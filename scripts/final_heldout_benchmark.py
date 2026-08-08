"""Guarded runner for the frozen final held-out benchmark.

Default execution never inspects the test directory. The irreversible first
final evaluation requires the exact ``--confirm-final-test`` flag.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, platform, sys, time, tracemalloc
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
CONFIG_PATH=ROOT/"configs/final_methodology_freeze.json"
OUTPUT_DIR=ROOT/"experiments/final_heldout_benchmark"
EXPECTED_FILES=("final_config_snapshot.json","environment_metadata.json","test_identity_manifest.csv",
 "payload_metadata.json","checkpoint_verification.json","per_sample_results.csv","per_condition_summary.csv",
 "paired_comparison_summary.csv","subband_summary.csv","fidelity_summary.json","computational_metrics.csv",
 "statistical_tests.csv","effect_sizes.csv","final_summary.json")

def load_config(path=CONFIG_PATH): return json.loads(Path(path).read_text(encoding="utf-8"))
def payload_from_config(cfg):
    bits=np.fromiter((int(x) for x in cfg["payload"]["bit_string"]),dtype=np.uint8)
    if len(bits)!=128 or int(bits.sum())!=64 or hashlib.sha256(bits.tobytes()).hexdigest()!=cfg["payload"]["sha256_uint8_bytes"]: raise RuntimeError("Frozen payload verification failed")
    return bits
def verify_checkpoint(cfg):
    p=ROOT/cfg["primary_cnn"]["checkpoint"]; digest=hashlib.sha256(p.read_bytes()).hexdigest()
    if p.stat().st_size!=cfg["primary_cnn"]["checkpoint_size_bytes"] or digest!=cfg["primary_cnn"]["checkpoint_sha256"]: raise RuntimeError("Frozen checkpoint identity mismatch")
    return p,digest
def apply_condition(image,name,cfg):
    from attacks.suite import jpeg_compression,resize_scale,crop_severity,reencode_jpeg
    if name=="clean": return image.copy()
    if name.startswith("jpeg"): return jpeg_compression(image,int(name[4:])).image
    if name.startswith("resize"): return resize_scale(image,int(name[6:])/100).image
    if name.startswith("crop_"): return crop_severity(image,name[5:],seed=cfg["attacks"]["crop"]["seed"]).image
    if name.startswith("reencode"): return reencode_jpeg(image,int(name[8:]),quality=85).image
    raise ValueError(name)
def classical_decisions(coefficients,delta=24.):
    from watermark.embedding import qim_extract_bit
    return np.asarray([qim_extract_bit(float(x),delta) for x in coefficients],np.uint8)
def selected_representation(image,cfg):
    from watermark.cnn_extraction import prepare_cnn_input_from_image
    from diagnose_run1_signal_localization import location_rows
    from stage3a_delta_calibration import features_from_map
    cmap=prepare_cnn_input_from_image(image,wavelet="haar"); locs=location_rows(cmap.shape[:2])
    co=np.asarray([cmap[x["row"],x["column"],x["channel"]] for x in locs],np.float32)
    return co,features_from_map(cmap,locs,cfg["watermark"]["delta"])
def timed(call,repetitions,logical_processors):
    wall=[];cpu=[];peaks=[];value=None
    for _ in range(repetitions):
        tracemalloc.start();w=time.perf_counter_ns();c=time.process_time_ns();value=call();cpu.append(time.process_time_ns()-c);wall.append(time.perf_counter_ns()-w);peaks.append(tracemalloc.get_traced_memory()[1]);tracemalloc.stop()
    mw=float(np.median(wall));mc=float(np.median(cpu))
    return value,{"wall_time_ns_median":mw,"process_time_ns_median":mc,"peak_python_bytes_median":float(np.median(peaks)),"cpu_utilization_percent":100*mc/(mw*logical_processors) if mw else 0.}
def holm_adjust(pvalues):
    p=np.asarray(pvalues,float); order=np.argsort(p); out=np.empty(len(p)); running=0.
    for rank,idx in enumerate(order): running=max(running,(len(p)-rank)*p[idx]);out[idx]=min(1.,running)
    return out
def rank_biserial(classical,cnn):
    from scipy.stats import rankdata
    d=np.asarray(classical)-np.asarray(cnn);d=d[d!=0]
    if not len(d): return 0.
    r=rankdata(abs(d),method="average");pos=r[d>0].sum();neg=r[d<0].sum();return float((pos-neg)/(pos+neg))
def statistical_rows(samples,conditions):
    from scipy.stats import wilcoxon
    rows=[]
    for condition in conditions:
        q=[x for x in samples if x["condition"]==condition];a=np.asarray([x["classical_ber"] for x in q]);b=np.asarray([x["cnn_ber"] for x in q]);d=a-b
        p=1. if np.all(d==0) else float(wilcoxon(a,b,zero_method="wilcox",alternative="two-sided",method="auto").pvalue)
        rows.append({"condition":condition,"n":len(q),"raw_p_value":p,"rank_biserial":rank_biserial(a,b),"mean_classical_minus_cnn_ber":float(d.mean())})
    adj=holm_adjust([x["raw_p_value"] for x in rows])
    for r,p in zip(rows,adj):r["holm_adjusted_p_value"]=float(p);r["reject_at_0.05"]=bool(p<=.05)
    return rows
def write_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summarize(rows,key):
    v=np.asarray([x[key] for x in rows],float);return {"mean":float(v.mean()),"median":float(np.median(v)),"std":float(v.std()),"minimum":float(v.min()),"maximum":float(v.max()),"perfect_count":int(np.sum(v==0)),"perfect_rate":float(np.mean(v==0))}
def run_final(cfg):
    # This is the first point at which the held-out image directory is touched.
    from tensorflow import keras
    import cv2, pywt, scipy, skimage, tensorflow as tf
    from dataset.preprocess import TARGET_SIZE,center_crop_square,resize_square
    from evaluation.metrics import compute_psnr,compute_ssim
    from watermark.extraction import extract_from_image
    from watermark.preprocessor import load_image
    from run_cnn_benchmark import embed_image
    test_dir=ROOT/"data/curated/test"; paths=sorted(test_dir.glob("*.png"))
    if len(paths)!=cfg["test_policy"]["images"]: raise RuntimeError(f"Expected 500 test images, found {len(paths)}")
    existing=[p for p in OUTPUT_DIR.iterdir() if p.name!="README.md"]
    if existing: raise FileExistsError(f"Refusing to overwrite final outputs: {existing}")
    checkpoint,digest=verify_checkpoint(cfg);model=keras.models.load_model(checkpoint,compile=False)
    if model.count_params()!=369 or list(model.input_shape)!=[None,128,4]: raise RuntimeError("Frozen architecture mismatch")
    payload=payload_from_config(cfg); reps=cfg["computational_benchmark"]["repetitions"];logical=cfg["computational_benchmark"]["machine"]["logical_processors"]
    dummy=np.zeros((1,128,4),np.float32)
    for _ in range(3): model.predict(dummy,verbose=0)
    samples=[];timings=[];fidelity=[];identities=[]
    for image_index,path in enumerate(paths):
        original=resize_square(center_crop_square(load_image(path)),TARGET_SIZE);watermarked=embed_image(original,payload,24.,"haar",42)
        fidelity.append({"image_index":image_index,"psnr_rgb_db":compute_psnr(original,watermarked),"ssim_rgb":compute_ssim(original,watermarked)})
        identities.append({"image_index":image_index,"image_filename":path.name,"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
        for condition in cfg["conditions"]:
            attacked=apply_condition(watermarked,condition,cfg) # exactly once, shared below
            def classical_e2e(): return extract_from_image(attacked,128,42,24.,"haar",target_subbands=("lh2","hl2"))[0]
            def cnn_e2e():
                _,feat=selected_representation(attacked,cfg);return (model.predict(feat[None],verbose=0)[0]>=.5).astype(np.uint8)
            cp,ct=timed(classical_e2e,reps,logical);npred,nt=timed(cnn_e2e,reps,logical)
            coefficients,features=selected_representation(attacked,cfg)
            _,cd=timed(lambda:classical_decisions(coefficients),reps,logical)
            _,nd=timed(lambda:(model.predict(features[None],verbose=0)[0]>=.5).astype(np.uint8),reps,logical)
            ce=cp!=payload;ne=npred!=payload
            samples.append({"image_index":image_index,"condition":condition,"classical_ber":float(ce.mean()),"cnn_ber":float(ne.mean()),
              "classical_lh2_ber":float(ce[:64].mean()),"classical_hl2_ber":float(ce[64:].mean()),"cnn_lh2_ber":float(ne[:64].mean()),"cnn_hl2_ber":float(ne[64:].mean()),
              "target_one_frequency":float(payload.mean()),"cnn_predicted_one_frequency":float(npred.mean()),"cnn_false_zero_count":int(np.sum((payload==1)&(npred==0))),"cnn_false_one_count":int(np.sum((payload==0)&(npred==1)))})
            for decoder,boundary,z in (("classical","end_to_end",ct),("cnn","end_to_end",nt),("classical","bit_decision_only",cd),("cnn","bit_decision_only",nd)):
                timings.append({"image_index":image_index,"condition":condition,"decoder":decoder,"boundary":boundary,**z})
    condition_rows=[];paired=[];subbands=[]
    for condition in cfg["conditions"]:
        q=[x for x in samples if x["condition"]==condition];cs=summarize(q,"classical_ber");ns=summarize(q,"cnn_ber");d=np.asarray([x["classical_ber"]-x["cnn_ber"] for x in q])
        condition_rows.append({"condition":condition,**{f"classical_{k}":v for k,v in cs.items()},**{f"cnn_{k}":v for k,v in ns.items()}})
        paired.append({"condition":condition,"cnn_better":int(np.sum(d>0)),"equal":int(np.sum(d==0)),"cnn_worse":int(np.sum(d<0)),"mean_classical_minus_cnn_ber":float(d.mean()),"relative_ber_reduction":float(d.mean()/cs["mean"]) if cs["mean"] else None})
        subbands.append({"condition":condition,**{k:float(np.mean([x[k] for x in q])) for k in ("classical_lh2_ber","classical_hl2_ber","cnn_lh2_ber","cnn_hl2_ber")}})
    attack_conditions=cfg["conditions"][1:];stat=statistical_rows(samples,attack_conditions)
    by_image=[]
    for i in range(500):
        q=[x for x in samples if x["image_index"]==i and x["condition"] in attack_conditions];by_image.append((np.mean([x["classical_ber"] for x in q]),np.mean([x["cnn_ber"] for x in q])))
    a=np.asarray([x[0] for x in by_image]);b=np.asarray([x[1] for x in by_image]);d=a-b
    from scipy.stats import wilcoxon
    ap=1. if np.all(d==0) else float(wilcoxon(a,b,zero_method="wilcox",alternative="two-sided",method="auto").pvalue)
    stat.append({"condition":"attack_macro_by_image","n":500,"raw_p_value":ap,"rank_biserial":rank_biserial(a,b),"mean_classical_minus_cnn_ber":float(d.mean()),"holm_adjusted_p_value":None,"reject_at_0.05":None})
    env={"os":platform.platform(),"python":platform.python_version(),"tensorflow":tf.__version__,"numpy":np.__version__,"pywavelets":pywt.__version__,"opencv":cv2.__version__,"scikit_image":skimage.__version__,"scipy":scipy.__version__,"thread_environment":{k:os.getenv(k) for k in ("TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS","OMP_NUM_THREADS")},"frozen_machine":cfg["computational_benchmark"]["machine"]}
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    (OUTPUT_DIR/"final_config_snapshot.json").write_text(json.dumps(cfg,indent=2),encoding="utf-8");(OUTPUT_DIR/"environment_metadata.json").write_text(json.dumps(env,indent=2),encoding="utf-8")
    (OUTPUT_DIR/"payload_metadata.json").write_text(json.dumps(cfg["payload"],indent=2),encoding="utf-8");(OUTPUT_DIR/"checkpoint_verification.json").write_text(json.dumps({"path":cfg["primary_cnn"]["checkpoint"],"sha256":digest,"size":checkpoint.stat().st_size,"parameters":model.count_params()},indent=2),encoding="utf-8")
    for name,data in (("test_identity_manifest",identities),("per_sample_results",samples),("per_condition_summary",condition_rows),("paired_comparison_summary",paired),("subband_summary",subbands),("computational_metrics",timings),("statistical_tests",stat),("effect_sizes",[{"condition":x["condition"],"rank_biserial":x["rank_biserial"]} for x in stat])):write_csv(OUTPUT_DIR/f"{name}.csv",data)
    fidelity_summary={k:float(v) for k,v in {"mean_psnr_rgb_db":np.mean([x["psnr_rgb_db"] for x in fidelity]),"mean_ssim_rgb":np.mean([x["ssim_rgb"] for x in fidelity])}.items()}
    (OUTPUT_DIR/"fidelity_summary.json").write_text(json.dumps({"summary":fidelity_summary,"per_image":fidelity},indent=2),encoding="utf-8")
    final={"conditions":condition_rows,"paired":paired,"statistics":stat,"fidelity":fidelity_summary,"configuration_sha256":hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()}
    (OUTPUT_DIR/"final_summary.json").write_text(json.dumps(final,indent=2),encoding="utf-8")
def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--confirm-final-test",action="store_true");p.add_argument("--validate-freeze",action="store_true");args=p.parse_args(argv)
    cfg=load_config();payload_from_config(cfg);verify_checkpoint(cfg)
    if args.validate_freeze:
        print("Freeze configuration, payload, and checkpoint identity verified; test directory was not accessed.");return 0
    if not args.confirm_final_test:
        print("REFUSED: final test access requires --confirm-final-test",file=sys.stderr);return 2
    run_final(cfg);return 0
if __name__=="__main__":raise SystemExit(main())
