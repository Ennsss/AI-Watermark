"""Stage 5A: zero-shot crop evaluation with synchronization diagnostics."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/"src"),str(ROOT/"scripts")]
from attacks.suite import crop_severity
from dataset.preprocess import TARGET_SIZE,center_crop_square,resize_square
from watermark.cnn_extraction import prepare_cnn_input_from_image
from watermark.extraction import extract_from_image
from watermark.preprocessor import load_image
from diagnose_run1_signal_localization import location_rows
from run3a_clean_transfer import distribution
from run_cnn_benchmark import embed_image
from stage2_jpeg_reencode_training import write_csv
from stage3a_delta_calibration import features_from_map

OUT=ROOT/"experiments/stage5a_zero_shot_crop"; S3=ROOT/"experiments/stage3a_delta_calibration"
S4=ROOT/"experiments/stage4a_zero_shot_resize"; VAL=ROOT/"data/curated/val"
CKPT=S3/"delta_24/best_seed_aware_delta24.keras"
CONDS=("clean","mild","moderate","severe"); SEED=0; DELTA=24.; BITS=128

def geometry(severity):
    ranges={"mild":(.05,.10),"moderate":(.20,.30),"severe":(.40,.50)}
    rng=np.random.default_rng(SEED); lo,hi=ranges[severity]; ratio=float(rng.uniform(lo,hi))
    rng=np.random.default_rng(SEED); linear=1-np.sqrt(1-ratio); h=w=512
    top=int(rng.uniform(0,linear)*h); bottom=int(rng.uniform(0,linear)*h)
    left=int(rng.uniform(0,linear)*w); right=int(rng.uniform(0,linear)*w)
    bottom=max(min(bottom,h-top-h//2),0); right=max(min(right,w-left-w//2),0)
    return {"severity":severity,"sampled_area_ratio":ratio,"linear_factor":float(linear),
            "top":top,"bottom":bottom,"left":left,"right":right,
            "retained_height":h-top-bottom,"retained_width":w-left-right}

def attack(im,c): return im.copy() if c=="clean" else crop_severity(im,c,seed=SEED).image
def coeffs(cmap,locs): return np.asarray([cmap[x["row"],x["column"],x["channel"]] for x in locs],np.float32)
def phase(x): return np.mod(x/DELTA,1.)
def cdist(a,b):
    d=np.abs(a-b); return np.minimum(d,1-d)
def cbits(x):
    p=phase(x); return (np.abs(p-.5)<np.minimum(p,1-p)).astype(np.uint8)
def patches(cmap,locs):
    pad=np.pad(cmap,((1,1),(1,1),(0,0)),mode="symmetric"); z=np.empty((BITS,3,3),np.float32)
    for q in locs: z[q["bit_index"]]=pad[q["row"]:q["row"]+3,q["column"]:q["column"]+3,q["channel"]]
    return z
def stats(v): return {"mean_ber":float(v.mean()),"median_ber":float(np.median(v)),"std_ber":float(v.std()),
    "minimum_ber":float(v.min()),"maximum_ber":float(v.max()),"perfect_payload_count":int(np.sum(v==0)),"perfect_payload_rate":float(np.mean(v==0))}
def label(v): return "EXCELLENT" if v<.01 else "GOOD" if v<.05 else "PARTIAL" if v<.20 else "POOR" if v<.40 else "NEAR RANDOM"

def main():
    from tensorflow import keras
    if OUT.exists() and any(OUT.iterdir()): raise FileExistsError(f"Refusing to overwrite {OUT}")
    model=keras.models.load_model(CKPT,compile=False)
    if model.count_params()!=369 or model.input_shape!=(None,128,4): raise RuntimeError("Frozen model mismatch")
    paths=sorted(VAL.glob("*.png"))[:100]
    expected=[r for r in csv.DictReader((S3/"payload_reproducibility_metadata.csv").open(encoding="utf-8")) if r["split"]=="validation"]
    total=800; X=np.empty((total,128,4),np.float32); Y=np.empty((total,128),np.uint8); C=np.empty_like(Y)
    CO=np.empty((total,128),np.float32); P=np.empty((total,128,3,3),np.float32); cids=np.empty(total,np.int8)
    rows=[]; payload_rows=[]; rng=np.random.default_rng(20260809); rid=base=0
    geom={c:geometry(c) for c in CONDS[1:]}; clean_locs=location_rows((128,128))
    locs_by={"clean":clean_locs}; map_shapes={"clean":[128,128]}
    for c in CONDS[1:]:
        g=geom[c]; shape=((g["retained_height"]+3)//4,(g["retained_width"]+3)//4)
        locs_by[c]=location_rows(shape); map_shapes[c]=list(shape)
    for ii,path in enumerate(paths,1):
        im=resize_square(center_crop_square(load_image(path)),TARGET_SIZE)
        if ii==1 or ii%25==0 or ii==100: print(f"[Stage 5A] {ii}/100 {path.name}",flush=True)
        for pi in range(2):
            bits=rng.integers(0,2,128,dtype=np.uint8); fp=hashlib.sha256(bits.tobytes()).hexdigest(); ex=expected[base]
            if fp!=ex["payload_fingerprint"] or path.name!=ex["image_filename"]: raise RuntimeError("Identity mismatch")
            payload_rows.append({"base_pair_index":base,"image_filename":path.name,"payload_index":pi,"payload_seed":20260809,"payload_fingerprint":fp})
            wm=embed_image(im,bits,DELTA,"haar",42)
            for cid,c in enumerate(CONDS):
                a=attack(wm,c); cmap=prepare_cnn_input_from_image(a,wavelet="haar"); locs=location_rows(cmap.shape[:2])
                if [(q["row"],q["column"],q["channel"]) for q in locs] != [(q["row"],q["column"],q["channel"]) for q in locs_by[c]]: raise RuntimeError("Location regeneration mismatch")
                X[rid]=features_from_map(cmap,locs,DELTA); CO[rid]=coeffs(cmap,locs); P[rid]=patches(cmap,locs); Y[rid]=bits;cids[rid]=cid
                C[rid],_=extract_from_image(a,128,42,DELTA,"haar",target_subbands=("lh2","hl2"))
                rows.append({"row_id":rid,"base_pair_index":base,"image_filename":path.name,"payload_index":pi,"payload_fingerprint":fp,"condition":c,
                             "attacked_height":a.shape[0],"attacked_width":a.shape[1],"dwt_subband_height":cmap.shape[0],"dwt_subband_width":cmap.shape[1]})
                rid+=1
            base+=1
    if not np.isfinite(X).all(): raise RuntimeError("Non-finite features")
    prob=np.asarray(model.predict(X,batch_size=32,verbose=0)); pred=(prob>=.5).astype(np.uint8)
    ce=C!=Y; ne=pred!=Y; cb=ce.mean(1); nb=ne.mean(1); conf=np.abs(prob-.5)*2
    cleanco=CO[cids==0]; cleanc=C[cids==0]
    summaries=[]; bitsout=[]; probs=[]; disp=[]; flips=[]
    for cid,c in enumerate(CONDS):
        m=cids==cid; e=ne[m]; ec=ce[m]; nbs=nb[m]; cbs=cb[m]; t=Y[m]; p=pred[m]; pr=prob[m]
        ns,cs=stats(nbs),stats(cbs); nmean=ns["mean_ber"]; cmean=cs["mean_ber"]
        summaries.append({"condition":c,**{f"classical_{k}":v for k,v in cs.items()},**{f"cnn_{k}":v for k,v in ns.items()},
          "absolute_cnn_improvement":cmean-nmean,"relative_cnn_ber_reduction":(cmean-nmean)/cmean,
          "classical_lh2_ber":float(ec[:,:64].mean()),"classical_hl2_ber":float(ec[:,64:].mean()),
          "cnn_lh2_ber":float(e[:,:64].mean()),"cnn_hl2_ber":float(e[:,64:].mean()),
          "cnn_better_count":int(np.sum(nbs<cbs)),"equal_count":int(np.sum(nbs==cbs)),"cnn_worse_count":int(np.sum(nbs>cbs)),"classification":label(nmean)})
        per=e.mean(0); tf=t.mean(0); pf=p.mean(0); worst=set(np.argsort(-per,kind="stable")[:10])
        for b in range(128): bitsout.append({"condition":c,"bit_index":b,"subband":"LH2" if b<64 else "HL2","cnn_ber":float(per[b]),
          "classical_ber":float(ec[:,b].mean()),"target_one_frequency":float(tf[b]),"predicted_one_frequency":float(pf[b]),
          "zero_ber":bool(per[b]==0),"always_zero":bool(pf[b]==0),"always_one":bool(pf[b]==1),"is_worst_10":b in worst})
        inc=e; cf=conf[m]
        probs.append({"condition":c,**{f"probability_{k}":v for k,v in distribution(pr.ravel()).items()},
          "fraction_0.45_to_0.55":float(np.mean((pr>=.45)&(pr<=.55))),"correct_prediction_confidence":float(cf[~inc].mean()),
          "incorrect_prediction_confidence":float(cf[inc].mean()) if inc.any() else None,"target_one_frequency":float(t.mean()),
          "predicted_one_frequency":float(p.mean()),"false_zero_count":int(np.sum(inc&(t==1))),"false_one_count":int(np.sum(inc&(t==0))),
          "zero_ber_bit_positions":int(np.sum(per==0)),"always_zero_positions":int(np.sum(pf==0)),"always_one_positions":int(np.sum(pf==1)),
          "worst_10_payload_positions":json.dumps(sorted(int(x) for x in worst))})
        d=np.abs(CO[m]-cleanco); ph=cdist(phase(CO[m]),phase(cleanco))
        disp.append({"condition":c,"mean_absolute_coefficient_displacement":float(d.mean()),"median_absolute_coefficient_displacement":float(np.median(d)),
          "p90_absolute_coefficient_displacement":float(np.quantile(d,.9)),"fraction_gt_delta_over_4":float(np.mean(d>6)),"fraction_gt_delta_over_2":float(np.mean(d>12)),
          "median_circular_phase_displacement":float(np.median(ph)),"mean_circular_phase_displacement":float(ph.mean())})
        flips.append({"condition":c,"classical_decision_flip_rate_vs_clean":float(np.mean(C[m]!=cleanc))})
    for i,r in enumerate(rows): r.update({"classical_ber":float(cb[i]),"cnn_ber":float(nb[i]),"cnn_mean_probability":float(prob[i].mean()),"cnn_mean_confidence":float(conf[i].mean())})
    # Approximate level-2 coefficient centers as 4x4 image-support centers.
    geometry_rows=[]; sync={}
    for c in CONDS[1:]:
        g=geom[c]; al=locs_by[c]; removed=[]; distances=[]
        for q0,q1 in zip(clean_locs,al):
            y0=(q0["row"]+.5)*4; x0=(q0["column"]+.5)*4
            removed.append(not(g["top"]<=y0<512-g["bottom"] and g["left"]<=x0<512-g["right"]))
            y1=g["top"]+(q1["row"]+.5)*4; x1=g["left"]+(q1["column"]+.5)*4
            distances.append(np.hypot(y1-y0,x1-x0))
        geometry_rows.append({"condition":c,**g,"attacked_dwt_shape":json.dumps(map_shapes[c]),
          "approx_fraction_original_selected_centers_removed":float(np.mean(removed)),"approx_mean_regenerated_center_mapping_distance_pixels":float(np.mean(distances)),
          "approx_median_regenerated_center_mapping_distance_pixels":float(np.median(distances)),"approx_fraction_regenerated_centers_within_4px_original":float(np.mean(np.asarray(distances)<=4))})
        sync[c]={"expected_coordinates_remain_aligned":False,"shape_dependent_seed_permutation_changes":True,
          "restoration_performed":False,"finding":"Cropped DWT shape changes the seed permutation; regenerated selected locations generally represent different original content."}
    # Conditional 3x3 non-deployable diagnostic.
    local_rows=[]; fixed_rows=[]; trigger=any(s["cnn_mean_ber"]>=.4 for s in summaries if s["condition"] in ("moderate","severe"))
    if trigger:
      for cid,c in enumerate(CONDS[1:],1):
        m=cids==cid; pp=P[m]; tt=Y[m]; cp=phase(cleanco); pp_phase=phase(pp)
        dist=cdist(pp_phase,cp[:,:,None,None]); flat=np.argmin(dist.reshape(200,128,9),2); chosen=np.take_along_axis(pp.reshape(200,128,9),flat[:,:,None],2)[:,:,0]
        local_rows.append({"condition":c,"oracle_clean_reference_3x3_ber":float(np.mean(cbits(chosen)!=tt)),
          "target_aware_any_correct_rate":float(np.mean(np.any(cbits(pp)==tt[:,:,None,None],axis=(2,3)))),
          "largest_oracle_offset_frequency":float(np.max(np.bincount(flat.ravel(),minlength=9))/flat.size)})
        for k,(dr,dc) in enumerate([(a,b) for a in (-1,0,1) for b in (-1,0,1)]):
          fixed_rows.append({"condition":c,"offset_row":dr,"offset_column":dc,"ber":float(np.mean(cbits(pp[:,:,dr+1,dc+1])!=tt))})
    cleanref={"stage3a_cnn":.0001953125,"current_cnn":summaries[0]["cnn_mean_ber"],"stage3a_classical":.1278515625,"current_classical":summaries[0]["classical_mean_ber"]}
    if abs(cleanref["current_cnn"]-cleanref["stage3a_cnn"])>1e-12 or abs(cleanref["current_classical"]-cleanref["stage3a_classical"])>1e-12: raise RuntimeError("Clean reproduction failed")
    mech={c:("SEVERE SYNCHRONIZATION LOSS" if next(s for s in summaries if s["condition"]==c)["cnn_mean_ber"]>=.4 else "PARTIAL SYNCHRONIZATION LOSS") for c in CONDS[1:]}
    summary={"condition_classifications":{s["condition"]:s["classification"] for s in summaries[1:]},"failure_mechanisms":mech,
      "crop_aware_bit_decision_training_justified":False,"synchronization_is_main_limitation":True,"local_oracle_triggered":trigger,
      "training_performed":False,"delta":24,"test_set_used":False}
    OUT.mkdir(parents=True)
    write_csv(OUT/"selected_validation_images.csv",[{"selection_order":i,"image_filename":p.name} for i,p in enumerate(paths,1)])
    for name,data in (("payload_reproducibility_metadata",payload_rows),("per_condition_summary",summaries),("per_sample_results",rows),("per_bit_diagnostics",bitsout),
      ("probability_diagnostics",probs),("coefficient_displacement_summary",disp),("classical_decision_flip_summary",flips),("geometric_mapping_summary",geometry_rows),("comparison_classical_cnn",summaries)):
      write_csv(OUT/f"{name}.csv",data)
    if trigger: write_csv(OUT/"local_crop_oracle_summary.csv",local_rows);write_csv(OUT/"fixed_offset_crop_results.csv",fixed_rows)
    config={"experiment":"Stage 5A zero-shot crop","evaluation_only":True,"conditions":list(CONDS),"crop_seed":SEED,"delta":24,"coefficient_seed":42,
      "checkpoint":str(CKPT.relative_to(ROOT)).replace("\\","/"),"threshold":.5,"test_set_used":False}
    implementation={"functions":["attacks.suite.crop_severity","attacks.suite.random_crop"],"library":"NumPy slicing","severity_ranges":{"mild":[.05,.10],"moderate":[.20,.30],"severe":[.40,.50]},
      "seed":SEED,"placement":"independent deterministic top/bottom/left/right edge removal","restoration":False,"interpolation":None,"exact_geometry":geom}
    verify={"validation_images":100,"base_pairs":200,"evaluation_rows":800,"filenames_match_stage3a":True,"payload_fingerprints_match_stage3a":True,
      "model_parameters":369,"model_input_shape":[None,128,4],"locations_per_image":128,"lh2":64,"hl2":64,"nan_count":0,"inf_count":0,"test_set_used":False}
    syncdoc={"diagnostic_only":True,"dwt_support_approximation":"Each level-2 coefficient center approximated as center of a 4x4 image region; boundary/wavelet support effects omitted.",
      "production_behavior":"Seeded locations are regenerated on each cropped DWT shape without coordinate correction.","conditions":sync}
    for name,obj in (("experiment_config",config),("crop_implementation",implementation),("data_identity_verification",verify),("synchronization_diagnostic",syncdoc),("summary_metrics",summary)):
      (OUT/f"{name}.json").write_text(json.dumps(obj,indent=2),encoding="utf-8")
    write_csv(OUT/"comparison_to_stage3a_clean.csv",[cleanref])
    print(json.dumps({"conditions":summaries,"displacement":disp,"flips":flips,"geometry":geometry_rows,"local":local_rows,"summary":summary},indent=2))
    return 0
if __name__=="__main__": raise SystemExit(main())
