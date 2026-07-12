from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait
from PIL import Image


FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
HEADLESS = os.getenv("HEADLESS", "false").lower() == "true"
TIMEOUT_SECONDS = int(os.getenv("SELENIUM_TIMEOUT", "30"))
LIVE_DELAY_SECONDS = float(os.getenv("SELENIUM_LIVE_DELAY", "0.8"))


@dataclass
class StepResult:
    name: str
    status: str
    details: str
    error: str | None = None


class SmokeTestRunner:
    def __init__(self) -> None:
        self.driver = self._build_driver()
        self.wait = WebDriverWait(self.driver, TIMEOUT_SECONDS)
        self.results: list[StepResult] = []
        self.ui_inventory: dict[str, dict[str, Any]] = {}
        self.upload_image_path = self._create_temp_png()
        self.live_step = 0

    def _build_driver(self) -> webdriver.Chrome:
        options = Options()
        if HEADLESS:
            options.add_argument("--headless=new")
        options.add_argument("--window-size=1440,1000")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.set_capability("goog:loggingPrefs", {"browser": "ALL"})
        return webdriver.Chrome(options=options)

    def _create_temp_png(self) -> str:
        temp_dir = Path(tempfile.gettempdir()) / "artifact_selenium"
        temp_dir.mkdir(parents=True, exist_ok=True)
        image_path = temp_dir / "sample_upload.png"
        image = Image.new("RGB", (32, 32), color=(120, 140, 220))
        image.save(image_path, format="PNG")
        return str(image_path)

    def add_result(self, name: str, status: str, details: str, error: str | None = None) -> None:
        self.results.append(StepResult(name=name, status=status, details=details, error=error))

    def _pause(self) -> None:
        if LIVE_DELAY_SECONDS > 0:
            time.sleep(LIVE_DELAY_SECONDS)

    def _live_log(self, status: str, action: str, details: str = "", error: str | None = None) -> None:
        if status == "RUN":
            self.live_step += 1
            prefix = f"[LIVE {self.live_step:02d}]"
        else:
            prefix = f"[LIVE {self.live_step:02d}]"

        message = f"{status} - {action}"
        if details:
            message = f"{message} - {details}"
        print(message)
        if error:
            print(f"{prefix} [ERROR] {error}")

    def _run_action(self, action: str, callback: Any, *, fatal: bool = True) -> bool:
        self._live_log("RUN", action)
        try:
            callback()
            self._live_log("PASS", action)
            self._pause()
            return True
        except Exception as exc:  # pylint: disable=broad-except
            self._live_log("FAIL", action, error=str(exc))
            self._pause()
            if fatal:
                raise
            return False

    def _inspect_elements(self, route_name: str, buttons: list[Any], text_inputs: list[Any], textareas: list[Any], selects: list[Any], file_inputs: list[Any], images: list[Any]) -> None:
        for index, button in enumerate(buttons, start=1):
            label = button.text.strip() or "<icon-button>"
            if not button.is_displayed():
                self._live_log(
                    "WARN",
                    f"Inspect {route_name} button #{index}",
                    f"Skipped hidden button: {label}",
                )
                self._pause()
                continue
            self._run_action(
                f"Inspect {route_name} button #{index}",
                lambda btn=button, txt=label: (
                    btn.is_displayed() or (_ for _ in ()).throw(Exception(f"Button not visible: {txt}"))
                ),
                fatal=False,
            )

        for index, input_el in enumerate(text_inputs, start=1):
            self._run_action(
                f"Inspect {route_name} text input #{index}",
                lambda el=input_el: (
                    el.is_enabled() or (_ for _ in ()).throw(Exception("Text input is disabled"))
                ),
                fatal=False,
            )

        for index, textarea in enumerate(textareas, start=1):
            self._run_action(
                f"Inspect {route_name} textarea #{index}",
                lambda el=textarea: (
                    el.is_enabled() or (_ for _ in ()).throw(Exception("Textarea is disabled"))
                ),
                fatal=False,
            )

        for index, select in enumerate(selects, start=1):
            self._run_action(
                f"Inspect {route_name} select #{index}",
                lambda el=select: (
                    el.is_enabled() or (_ for _ in ()).throw(Exception("Select is disabled"))
                ),
                fatal=False,
            )

        for index, file_input in enumerate(file_inputs, start=1):
            self._run_action(
                f"Inspect {route_name} file upload #{index}",
                lambda el=file_input: (
                    el.is_enabled() or (_ for _ in ()).throw(Exception("File input is disabled"))
                ),
                fatal=False,
            )

        for index, image in enumerate(images, start=1):
            if not image.is_displayed():
                self._live_log(
                    "WARN",
                    f"Inspect {route_name} image #{index}",
                    "Skipped hidden image",
                )
                self._pause()
                continue
            self._run_action(
                f"Inspect {route_name} image #{index}",
                lambda el=image: (
                    el.is_displayed() or (_ for _ in ()).throw(Exception("Image is not displayed"))
                ),
                fatal=False,
            )

    def route_inventory(self, route_name: str) -> None:
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        text_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='text']")
        textareas = self.driver.find_elements(By.TAG_NAME, "textarea")
        selects = self.driver.find_elements(By.TAG_NAME, "select")
        file_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='file']")
        images = self.driver.find_elements(By.TAG_NAME, "img")

        self._inspect_elements(route_name, buttons, text_inputs, textareas, selects, file_inputs, images)

        self.ui_inventory[route_name] = {
            "buttons": [b.text.strip() or "<icon-button>" for b in buttons],
            "text_inputs": len(text_inputs),
            "textareas": len(textareas),
            "selects": len(selects),
            "file_uploads": len(file_inputs),
            "images": len(images),
        }

    def check_console_errors(self, where: str) -> None:
        errors = []
        try:
            for entry in self.driver.get_log("browser"):
                level = str(entry.get("level", "")).upper()
                msg = str(entry.get("message", ""))
                if level in {"SEVERE", "ERROR"} and "favicon.ico" not in msg:
                    errors.append(msg)
        except WebDriverException:
            # Some drivers may not support browser logs in all environments.
            pass

        if errors:
            self.add_result(
                name=f"Console errors on {where}",
                status="FAIL",
                details=f"Found {len(errors)} browser console errors",
                error="\n".join(errors[:5]),
            )
        else:
            self.add_result(
                name=f"Console errors on {where}",
                status="PASS",
                details="No severe browser console errors detected",
            )

    def go(self, path: str, route_name: str, wait_selector: tuple[str, str]) -> None:
        def _navigate() -> None:
            self.driver.get(f"{FRONTEND_URL}{path}")
            self.wait.until(EC.presence_of_element_located(wait_selector))

        self._run_action(f"Open route {path}", _navigate)
        self.route_inventory(route_name)
        self.check_console_errors(route_name)

    def click_first(self, selector: str) -> bool:
        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
        if not elements:
            self._live_log("WARN", f"Click first by selector {selector}", "No matching element found")
            return False

        def _click() -> None:
            self.wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            elements[0].click()

        self._run_action(f"Click first by selector {selector}", _click)
        return True

    def run(self) -> int:
        try:
            self._run_steps()
        except Exception as exc:  # pylint: disable=broad-except
            self.add_result(
                name="Unexpected runner failure",
                status="FAIL",
                details="Unhandled exception stopped the smoke test",
                error=str(exc),
            )
        finally:
            self.driver.quit()

        return self._print_and_persist_report()

    def _run_steps(self) -> None:
        # Dashboard
        self.go("/", "dashboard", (By.XPATH, "//h1[contains(., 'Dashboard')]"))

        # Register flow
        self.go("/register", "register", (By.XPATH, "//h1[contains(., 'Register Artwork')]"))

        self._run_action(
            "Type artwork title",
            lambda: self.driver.find_element(By.ID, "title").send_keys("Sample Artwork"),
        )
        self._run_action(
            "Type creator name",
            lambda: self.driver.find_element(By.ID, "creator_name").send_keys("Test"),
        )
        self._run_action(
            "Type artwork notes",
            lambda: self.driver.find_element(By.ID, "notes").send_keys("This is a test run"),
        )
        self._run_action(
            "Upload register image",
            lambda: self.driver.find_element(By.CSS_SELECTOR, "#file-input").send_keys(self.upload_image_path),
        )
        self._run_action(
            "Click Register & Embed Watermark button",
            lambda: self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click(),
        )

        self._run_action(
            "Wait for register success message",
            lambda: self.wait.until(EC.presence_of_element_located((By.XPATH, "//h2[contains(., 'Artwork Registered Successfully')]"))),
        )
        self.route_inventory("register_success")
        self.add_result(
            name="Register artwork",
            status="PASS",
            details="Artwork registration and watermark embedding completed",
        )
        self.check_console_errors("register success")

        # View artwork detail from success page
        self._run_action(
            "Click View Artwork Record button",
            lambda: self.driver.find_element(By.XPATH, "//button[contains(., 'View Artwork Record')]" ).click(),
        )
        self._run_action(
            "Wait for artwork detail preview",
            lambda: self.wait.until(EC.presence_of_element_located((By.XPATH, "//h2[contains(., 'Embedded Watermark Preview')]"))),
        )
        self.route_inventory("artwork_detail")
        self._run_action(
            "Open detail unregister modal",
            lambda: self.driver.find_element(By.XPATH, "//button[contains(., 'Unregister Artwork')]").click(),
        )
        self._run_action(
            "Wait for detail unregister modal",
            lambda: self.wait.until(EC.presence_of_element_located((By.ID, "unregister-title"))),
        )
        self._run_action(
            "Cancel detail unregister modal",
            lambda: self.driver.find_element(By.XPATH, "//div[contains(@class,'unregister-modal')]//button[contains(., 'Cancel')]").click(),
        )
        self._run_action(
            "Wait for detail unregister modal close",
            lambda: self.wait.until(EC.invisibility_of_element_located((By.ID, "unregister-title"))),
        )
        self.add_result(
            name="Artwork details",
            status="PASS",
            details="Artwork details page loaded from registration result",
        )
        self.check_console_errors("artwork detail")

        # Back to artworks list
        self._run_action(
            "Click back button on artwork detail",
            lambda: self.driver.find_element(By.XPATH, "//button[contains(., 'Back to artworks') or contains(., 'Back to Registry')]" ).click(),
        )
        self._run_action(
            "Wait for My Artworks page",
            lambda: self.wait.until(EC.presence_of_element_located((By.XPATH, "//h1[contains(., 'My Artworks')]"))),
        )
        self.route_inventory("artworks")
        self._run_action(
            "Open artworks unregister modal",
            lambda: self.driver.find_element(By.CSS_SELECTOR, "button.artwork-trash-button").click(),
        )
        self._run_action(
            "Wait for artworks unregister modal",
            lambda: self.wait.until(EC.presence_of_element_located((By.ID, "card-unregister-title"))),
        )
        self._run_action(
            "Cancel artworks unregister modal",
            lambda: self.driver.find_element(By.XPATH, "//div[contains(@class,'unregister-modal')]//button[contains(., 'Cancel')]").click(),
        )
        self._run_action(
            "Wait for artworks unregister modal close",
            lambda: self.wait.until(EC.invisibility_of_element_located((By.ID, "card-unregister-title"))),
        )
        self.check_console_errors("artworks")

        # Verify flow via artworks page button or direct route fallback
        clicked_verify = self.click_first(".artwork-card-footer .btn-primary")
        if not clicked_verify:
            self._run_action(
                "Fallback open verify route",
                lambda: self.driver.get(f"{FRONTEND_URL}/verify"),
            )
        self._run_action(
            "Wait for Verify Image page",
            lambda: self.wait.until(EC.presence_of_element_located((By.XPATH, "//h1[contains(., 'Verify Image')]"))),
        )
        self.route_inventory("verify")

        artwork_select = Select(self.driver.find_element(By.ID, "artwork-select"))
        if len(artwork_select.options) < 2:
            self.add_result(
                name="Verify image",
                status="FAIL",
                details="No artwork options available in verify dropdown",
            )
        else:
            self._run_action(
                "Select artwork in verify dropdown",
                lambda: artwork_select.select_by_index(1),
            )
            self._run_action(
                "Upload suspected image",
                lambda: self.driver.find_element(By.CSS_SELECTOR, "#file-input").send_keys(self.upload_image_path),
            )
            self._run_action(
                "Click Verify Watermark button",
                lambda: self.driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click(),
            )
            self._run_action(
                "Wait for verify result card",
                lambda: self.wait.until(EC.presence_of_element_located((By.XPATH, "//h2[contains(., 'Verified Match') or contains(., 'Partial') or contains(., 'No Valid Watermark')]"))),
            )
            self.route_inventory("verify_result")
            self.add_result(
                name="Verify image",
                status="PASS",
                details="Verification flow completed and result shown",
            )
            self.check_console_errors("verify result")

            if self.click_first("button.btn-outline"):
                self._run_action(
                    "Wait for Verify Image page after reset",
                    lambda: self.wait.until(EC.presence_of_element_located((By.XPATH, "//h1[contains(., 'Verify Image')]"))),
                )

        # History flow
        self.go("/history", "history", (By.XPATH, "//h1[contains(., 'Verification History')]"))

        history_card_xpath = "//article[contains(@class,'history-card')]"
        history_archive_button_xpath = "(//article[contains(@class,'history-card')])[1]//button[contains(@title, 'Archive verification')]"
        history_cards_before_archive = len(self.driver.find_elements(By.XPATH, history_card_xpath))

        if history_cards_before_archive > 0:
            self._run_action(
                "Open archive verification modal",
                lambda: (
                    self.driver.execute_script(
                        "arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});",
                        self.driver.find_element(By.XPATH, history_archive_button_xpath),
                    ),
                    self.wait.until(EC.element_to_be_clickable((By.XPATH, history_archive_button_xpath))),
                    self.driver.execute_script(
                        "arguments[0].click();",
                        self.driver.find_element(By.XPATH, history_archive_button_xpath),
                    ),
                ),
            )
            self._run_action(
                "Confirm archive verification",
                lambda: self.driver.find_element(By.XPATH, "//button[contains(., 'Archive')]").click(),
            )
            self._run_action(
                "Reload history after archive",
                lambda: self.driver.get(f"{FRONTEND_URL}/history"),
            )
            self._run_action(
                "Wait for history page after archive",
                lambda: self.wait.until(EC.presence_of_element_located((By.XPATH, "//h1[contains(., 'Verification History')]"))),
            )
            archive_reflected = self._run_action(
                "Wait for persisted verification count decrease",
                lambda: self.wait.until(
                    lambda _driver: len(self.driver.find_elements(By.XPATH, history_card_xpath)) < history_cards_before_archive
                ),
                fatal=False,
            )
            if archive_reflected:
                self.add_result(
                    name="Archive verification history",
                    status="PASS",
                    details="Archived one verification history entry from the history page",
                )
            else:
                self.add_result(
                    name="Archive verification history",
                    status="FAIL",
                    details="Archive action did not remove a history entry; check backend /api/verifications/{id}/archive",
                )
        else:
            self.add_result(
                name="Archive verification history",
                status="WARN",
                details="No verification history cards were available to archive",
            )

        view_detail_clicked = self.click_first(".history-card-footer .btn-outline")
        if view_detail_clicked:
            self._run_action(
                "Wait for verification detail page",
                lambda: self.wait.until(EC.presence_of_element_located((By.XPATH, "//h1[contains(., 'Verification Details')]"))),
            )
            self.route_inventory("history_detail")
            self.add_result(
                name="History details",
                status="PASS",
                details="Opened a verification detail record",
            )
            self.check_console_errors("history detail")
        else:
            self.add_result(
                name="History details",
                status="WARN",
                details="No verification cards found to open details",
            )

        # Cleanup flow: delete one test artwork at the end to validate unregister action.
        self.go("/artworks", "artworks_cleanup", (By.XPATH, "//h1[contains(., 'My Artworks')]"))
        test_artwork_card_xpath = (
            "//article[contains(@class,'artwork-card')][.//h2[contains(@class,'artwork-card-title') and @title='Sample Artwork']]"
        )
        before_count = len(self.driver.find_elements(By.XPATH, test_artwork_card_xpath))

        if before_count == 0:
            self.add_result(
                name="Delete test artwork",
                status="WARN",
                details="No 'Sample Artwork' card found to unregister during cleanup",
            )
        else:
            self._run_action(
                "Open cleanup unregister modal",
                lambda: self.driver.find_element(
                    By.XPATH,
                    f"({test_artwork_card_xpath}//button[contains(@class,'artwork-trash-button')])[1]",
                ).click(),
            )
            self._run_action(
                "Wait for cleanup unregister modal",
                lambda: self.wait.until(EC.presence_of_element_located((By.ID, "card-unregister-title"))),
            )
            self._run_action(
                "Confirm cleanup unregister",
                lambda: self.driver.find_element(
                    By.XPATH,
                    "//div[contains(@class,'unregister-modal')]//button[contains(@class,'btn-danger') and contains(., 'Unregister Artwork')]",
                ).click(),
            )
            self._run_action(
                "Wait for cleanup unregister modal close",
                lambda: self.wait.until(EC.invisibility_of_element_located((By.ID, "card-unregister-title"))),
            )
            self._run_action(
                "Wait for test artwork count decrease",
                lambda: self.wait.until(
                    lambda _driver: len(self.driver.find_elements(By.XPATH, test_artwork_card_xpath)) < before_count
                ),
            )
            self.add_result(
                name="Delete test artwork",
                status="PASS",
                details="Unregistered one 'Sample Artwork' entry during cleanup",
            )
            self.check_console_errors("artworks cleanup")

        self.add_result(
            name="Navigation coverage",
            status="PASS",
            details="Visited dashboard, register, artwork details, artworks, verify, and history",
        )

    def _print_and_persist_report(self) -> int:
        total = len(self.results)
        passed = len([r for r in self.results if r.status == "PASS"])
        failed = len([r for r in self.results if r.status == "FAIL"])
        warned = len([r for r in self.results if r.status == "WARN"])

        report = {
            "frontend_url": FRONTEND_URL,
            "generated_at_epoch": time.time(),
            "summary": {
                "total_steps": total,
                "passed": passed,
                "failed": failed,
                "warned": warned,
            },
            "ui_inventory": self.ui_inventory,
            "steps": [asdict(r) for r in self.results],
        }

        report_path = Path.cwd() / "test_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(" ")
        print("TEST REPORT")
        print(" ")
        print(
            f"Summary: total={total}, passed={passed}, failed={failed}, warned={warned}"
        )
        print("\nStep results:")
        for index, result in enumerate(self.results, start=1):
            print(f"{result.status} - {result.name} - {result.details}")
            if result.error:
                print(f"    error: {result.error}")

        return 0 if failed == 0 else 1


def main() -> int:
    runner = SmokeTestRunner()
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
