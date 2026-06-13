#!/usr/bin/env python3
"""
测试脚本：对三张示例图像分别调用分类、分割、报告生成接口，验证响应格式。
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:9989"
IMAGES = {
    "classification": "/kaggle/working/LLaVA-Rad/class_example.png",
    "segmentation": "/kaggle/working/LLaVA-Rad/segment_example.png",
    "report": "/kaggle/working/LLaVA-Rad/report_example.jpg",
}

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {detail}" if detail else f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ FAIL: {name}" + (f" — {detail}" if detail else ""))


def test_health():
    print("\n" + "=" * 60)
    print("  1. Health Check")
    print("=" * 60)
    r = requests.get(f"{BASE_URL}/api/health")
    check("status 200", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    check("status field", data.get("status") == "ok", f"body={data}")
    check("device field", "device" in data, str(data.get("device")))


def test_classification():
    print("\n" + "=" * 60)
    print("  2. Classification — POST /api/inference/classification")
    print("=" * 60)

    path = IMAGES["classification"]
    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/api/inference/classification",
            files={"image": (path.rsplit("/", 1)[-1], f, "image/png")},
        )

    check("status 200", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    print(f"  Response: {json.dumps(data, indent=2)}")

    check("final_result present", "final_result" in data)
    check("normal_prob present", "normal_prob" in data)
    check("lung_opacity_prob present", "lung_opacity_prob" in data)
    check("nlo_nn_prob present", "nlo_nn_prob" in data)

    probs = [data.get("normal_prob", 0), data.get("lung_opacity_prob", 0), data.get("nlo_nn_prob", 0)]
    total = sum(probs)
    check("probabilities ≈ 1.0", abs(total - 1.0) < 0.01, f"sum={total:.4f}")
    check("final_result matches max prob",
          data["final_result"] == max(
              [("Normal", data.get("normal_prob", 0)),
               ("Lung_Opacity", data.get("lung_opacity_prob", 0)),
               ("No_Lung_Opacity_Not_Normal", data.get("nlo_nn_prob", 0))],
              key=lambda x: x[1],
          )[0],
          f"final_result={data.get('final_result')}")


def test_segmentation():
    print("\n" + "=" * 60)
    print("  3. Segmentation — POST /api/inference/segmentation")
    print("=" * 60)

    path = IMAGES["segmentation"]
    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/api/inference/segmentation",
            files={"image": (path.rsplit("/", 1)[-1], f, "image/png")},
        )

    check("status 200", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    print(f"  Response: {json.dumps(data, indent=2)}")

    check("output_url present", "output_url" in data)
    output_url = data.get("output_url", "")
    check("output_url is http/https", output_url.startswith("http"), f"url={output_url}")

    # 验证返回的 URL 可访问
    if output_url.startswith("http"):
        try:
            img_r = requests.head(output_url, timeout=10)
            check("mask URL accessible", img_r.status_code == 200, f"status={img_r.status_code}")
        except Exception as e:
            check("mask URL accessible", False, f"error: {e}")


def test_report():
    print("\n" + "=" * 60)
    print("  4. Report Generation — POST /api/inference/report")
    print("=" * 60)

    path = IMAGES["report"]
    with open(path, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/api/inference/report",
            files={"image": (path.rsplit("/", 1)[-1], f, "image/jpeg")},
        )

    check("status 200", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    print(f"  Response: {json.dumps(data, indent=2)[:500]}...")

    check("output_report present", "output_report" in data)
    report = data.get("output_report", "")
    check("output_report is non-empty string", isinstance(report, str) and len(report) > 10,
          f"length={len(report)}")
    # 检查报告包含英文医学描述关键词
    has_medical = any(word in report.lower() for word in ["heart", "lung", "normal", "pleural", "effusion", "pneumothorax", "mediastinal"])
    check("contains medical terms", has_medical, f"report preview: {report[:120]}...")


def test_records():
    print("\n" + "=" * 60)
    print("  5. Records — GET /api/records/*")
    print("=" * 60)

    for task in ["classification", "segmentation", "report"]:
        r = requests.get(f"{BASE_URL}/api/records/{task}")
        check(f"GET /api/records/{task} status 200", r.status_code == 200)
        data = r.json()
        check(f"  returns list", isinstance(data, list), f"items={len(data)}")
        if data:
            print(f"    Latest {task} record: {json.dumps(data[0], indent=4)[:300]}")


def test_auth():
    print("\n" + "=" * 60)
    print("  6. Auth — POST /api/auth/register & /api/auth/login")
    print("=" * 60)

    # Register
    r = requests.post(f"{BASE_URL}/api/auth/register", json={"username": "testuser", "password": "test123"})
    print(f"  Register: status={r.status_code}, body={r.json()}")

    # Login
    r = requests.post(f"{BASE_URL}/api/auth/login", json={"username": "testuser", "password": "test123"})
    check("login status 200", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    check("token present", "token" in data)
    check("user present", "user" in data and "username" in data.get("user", {}))


if __name__ == "__main__":
    # 等待服务就绪
    print("Waiting for server to be ready...")
    for i in range(60):
        try:
            r = requests.get(f"{BASE_URL}/api/health", timeout=3)
            if r.status_code == 200:
                print(f"Server is ready (attempt {i+1})")
                break
        except requests.ConnectionError:
            pass
        time.sleep(5)
    else:
        print("ERROR: Server did not become ready within 5 minutes")
        sys.exit(1)

    test_health()
    test_classification()
    test_segmentation()
    test_report()
    test_records()
    test_auth()

    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed out of {passed+failed}")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
