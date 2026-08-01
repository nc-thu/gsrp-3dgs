"""Build an offline Chinese evidence report from real-CUDA precision results."""
from __future__ import annotations

import base64
import csv
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _summarize(rows: list[dict[str, str]], key: str = "choice") -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    result = []
    for name, group in grouped.items():
        delta_psnr = [float(row["delta_psnr_vs_standard"]) for row in group]
        delta_ssim = [float(row["delta_ssim_vs_standard"]) for row in group]
        direct = [float(row["psnr_standard_ref"]) for row in group]
        result.append({
            "profile": name,
            "cameras": len(group),
            "mean_delta_psnr": _mean(delta_psnr),
            "worst_delta_psnr": min(delta_psnr),
            "mean_delta_ssim": _mean(delta_ssim),
            "mean_direct_psnr": _mean(direct),
            "worst_direct_psnr": min(direct),
            "saturations": sum(int(row["saturations"]) for row in group),
            "max_abs_error": max(float(row["max_abs_error_vs_standard"]) for row in group),
        })
    return result


def _image_data(path: Path) -> str:
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _bar(value: float, maximum: float, color: str) -> str:
    width = max(1.0, min(100.0, value / maximum * 100.0))
    return f'<div class="bar"><i style="width:{width:.2f}%;background:{color}"></i></div>'


def build_report(results: Path, output: Path | None = None) -> dict[str, object]:
    output = output or results / "report.html"
    calibration = _read_csv(results / "camera0_input_sweep.csv")
    selected = json.loads((results / "candidate_formats.json").read_text(encoding="utf-8"))["formats"]
    camera0_ranges = json.loads((results / "ranges.json").read_text(encoding="utf-8"))
    heldout_ranges = json.loads((results / "heldout_ranges.json").read_text(encoding="utf-8"))
    full_rows = _read_csv(results / "heldout_input_validation.csv")
    control_rows = _read_csv(results / "controls_v2" / "heldout_input_validation.csv")
    control_summary = _summarize(control_rows)

    single_summary = []
    for signal in selected:
        for choice in ("minimum", "safe"):
            group = [
                row for row in full_rows
                if row["signal"] == signal and row["choice"] == choice
            ]
            values = _summarize(group)
            if values:
                values[0]["signal"] = signal
                values[0]["choice"] = choice
                single_summary.append(values[0])

    summary = {
        "schema": "gs-raster-precision-v2-report-1",
        "dataset": "T&T-107k",
        "allocator": "ATAE-SNUGBOX",
        "reference": "standard renderCUDA",
        "calibration_camera": 0,
        "heldout_cameras": 37,
        "camera0_formats": selected,
        "joint_profiles": control_summary,
        "decision": {
            "simulator_contract_valid": True,
            "camera0_profile_generalizes": False,
            "preliminary_tt_candidate": "diagnostic_balanced",
            "frozen_cross_scene_profile": False,
        },
    }
    (results / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (results / "profile_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(control_summary[0]))
        writer.writeheader()
        writer.writerows(control_summary)

    labels = {
        "mean_x": "投影中心 x",
        "mean_y": "投影中心 y",
        "conic_a": "二次型 A",
        "conic_b": "二次型 B",
        "conic_c": "二次型 C",
        "opacity": "不透明度",
        "rgb": "高斯 RGB",
    }
    format_rows = []
    for signal, choices in selected.items():
        c0 = camera0_ranges[signal]
        ho = heldout_ranges[signal]
        format_rows.append(
            "<tr>"
            f"<td>{labels[signal]}</td><td><code>{choices['minimum']['qname']}</code></td>"
            f"<td><code>{choices['safe']['qname']}</code></td>"
            f"<td>{c0['minimum']:.5g} ～ {c0['maximum']:.5g}</td>"
            f"<td>{ho['minimum']:.5g} ～ {ho['maximum']:.5g}</td>"
            "</tr>"
        )

    profile_order = [
        "fallback_wide", "diagnostic_compact", "diagnostic_balanced",
        "diagnostic_quality_safe", "convergence_control",
    ]
    summaries = {row["profile"]: row for row in control_summary}
    profile_rows = []
    for name in profile_order:
        row = summaries[name]
        status = (
            "控制组" if name == "convergence_control"
            else "T&T 候选" if name.startswith("diagnostic_")
            else "失败诊断"
        )
        profile_rows.append(
            "<tr>"
            f"<td><code>{name}</code><small>{status}</small></td>"
            f"<td>{row['mean_delta_psnr']:+.4f} dB</td>"
            f"<td>{row['worst_delta_psnr']:+.4f} dB</td>"
            f"<td>{row['mean_direct_psnr']:.2f} dB{_bar(row['mean_direct_psnr'], 90, '#5b8def')}</td>"
            f"<td>{row['worst_direct_psnr']:.2f} dB</td>"
            f"<td>{row['saturations']:,}</td>"
            "</tr>"
        )

    single_rows = []
    for row in single_summary:
        single_rows.append(
            "<tr>"
            f"<td>{labels[row['signal']]}</td><td>{row['choice']}</td>"
            f"<td>{row['mean_delta_psnr']:+.4f}</td><td>{row['worst_delta_psnr']:+.4f}</td>"
            f"<td>{row['worst_direct_psnr']:.2f}</td><td>{row['saturations']:,}</td>"
            "</tr>"
        )

    standard = _image_data(results / "images" / "standard_camera_00000.png")
    mean_image = _image_data(results / "images" / "mean_x_minimum_camera_00000.png")
    conic_image = _image_data(results / "images" / "conic_c_minimum_camera_00000.png")
    css = """
    :root{--ink:#172033;--muted:#667085;--blue:#2563eb;--green:#0f9d76;--amber:#d97706;
    --red:#c2413b;--paper:#f5f7fb;--card:#fff;--line:#dce3ef}
    *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
    font:16px/1.7 "Microsoft YaHei","Noto Sans SC",sans-serif}
    main{max-width:1180px;margin:auto;padding:34px 22px 80px}.hero{padding:34px;border-radius:24px;
    color:#fff;background:linear-gradient(135deg,#16213e,#244a84 60%,#2877a8)}
    h1{font-size:34px;line-height:1.25;margin:0 0 12px}h2{margin:42px 0 14px;font-size:27px}
    h3{margin:10px 0 5px}.lede{font-size:19px;max-width:900px}.grid{display:grid;gap:16px}
    .g3{grid-template-columns:repeat(3,1fr)}.g2{grid-template-columns:repeat(2,1fr)}
    .card{background:var(--card);border:1px solid var(--line);border-radius:17px;padding:20px;
    box-shadow:0 6px 20px #273c5b10}.good{border-top:5px solid var(--green)}
    .warn{border-top:5px solid var(--amber)}.bad{border-top:5px solid var(--red)}
    .big{font-size:28px;font-weight:800}.muted,small{color:var(--muted)}small{display:block}
    table{width:100%;border-collapse:collapse;background:white;border-radius:15px;overflow:hidden}
    th,td{padding:12px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
    th{background:#edf3fc;font-size:14px}code{background:#eef2f8;padding:2px 6px;border-radius:6px}
    .flow{display:flex;gap:10px;align-items:stretch;flex-wrap:wrap}.node{flex:1;min-width:130px;
    background:#fff;border:2px solid #c8d6ed;padding:15px;border-radius:14px;text-align:center}
    .arrow{align-self:center;font-size:25px;color:var(--blue)}.bar{height:7px;background:#e9eef6;
    border-radius:10px;margin-top:6px}.bar i{display:block;height:100%;border-radius:10px}
    .formula{font-size:20px;text-align:center;background:#13213b;color:#fff;border-radius:14px;padding:18px}
    .shots img{width:100%;border-radius:10px;border:1px solid var(--line)}
    .tag{display:inline-block;padding:3px 10px;border-radius:99px;background:#dff6ed;color:#087454;
    font-weight:700}.callout{border-left:5px solid var(--amber);background:#fff8e8;padding:18px;border-radius:10px}
    @media(max-width:780px){.g2,.g3{grid-template-columns:1fr}.arrow{display:none}h1{font-size:28px}
    th,td{font-size:13px;padding:8px 6px}main{padding:18px 10px 60px}}
    """
    body = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>GSRP v2：输入量化结果</title>
    <style>{css}</style></head><body><main>
    <section class="hero"><span class="tag">真实 CUDA · T&T-107k · 2026-07-24</span>
    <h1>3DGS 光栅化输入到底该用多少 bit？</h1>
    <p class="lede">这轮把 x/y、二次型 A/B/C、opacity 和 RGB 分开量化，再在 37 张未参与选格式的相机上复查。
    结论不是“已经找到最终位宽”，而是：仿真链路可信；camera 0 单独校准不够；一套较稳妥的 T&T 候选已经出现，
    但必须经过跨场景校准后才能成为社区推荐格式。</p></section>

    <section class="grid g3" style="margin-top:18px">
      <article class="card good"><h3>仿真器通过控制实验</h3><div class="big">82.75 dB</div>
      <p>高精度定点控制组与标准 renderer 的平均直接 PSNR；最差仍有 57.92 dB，且零饱和。</p></article>
      <article class="card bad"><h3>camera 0 格式不能直接推广</h3><div class="big">1,810,000+</div>
      <p>原 minimum/safe 联合格式在 held-out 相机中的饱和量级。主要原因是坐标范围扩张和 RGB 超过 4。</p></article>
      <article class="card warn"><h3>当前推荐动作</h3><div class="big">先校准，再冻结</div>
      <p><code>diagnostic_balanced</code> 可作为下一轮八场景候选，不应立即写成最终硬件规范。</p></article>
    </section>

    <h2>1. 仿真器实际在算什么</h2>
    <div class="flow">
      <div class="node"><b>FP16 预处理</b><br>投影、ATAE 分配和深度顺序保持不变</div><div class="arrow">→</div>
      <div class="node"><b>七个量化入口</b><br>x/y、A/B/C、opacity、RGB</div><div class="arrow">→</div>
      <div class="node"><b>标准逐像素二次型</b><br>没有 SARD recurrence</div><div class="arrow">→</div>
      <div class="node"><b>EXP + blending</b><br>本轮维持 float，隔离输入误差</div><div class="arrow">→</div>
      <div class="node"><b>完整图像</b><br>和标准 renderCUDA、GT 同时比较</div>
    </div>
    <p class="formula">power = −0.5 × (A·dx² + C·dy²) − B·dx·dy</p>
    <div class="callout"><b>为什么 mean_x / mean_y 用 tile-local？</b>
    先减去 16×16 tile 左上角，再量化。这样保存的是“高斯中心相对本 tile 的位置”，不会把大量 bit 浪费在全图坐标上。
    但高斯中心可以落在 tile 外很远，因此 local 不等于数值一定只在 0～15。</div>

    <h2>2. 三道质量门，缺一不可</h2>
    <section class="grid g3">
      <article class="card"><h3>① PSNR@GT</h3><p>回答“最终图像整体好不好”。量化图偶尔会碰巧更接近 GT，所以这个指标可能变好。</p></article>
      <article class="card"><h3>② fixed vs standard</h3><p>回答“硬件近似改了多少”。两张都不好的图也可能互相很像，反过来也一样，因此它不能替代 GT 指标。</p></article>
      <article class="card"><h3>③ 饱和/溢出</h3><p>回答“数值是否真的装得下”。即使 PSNR 暂时没坏，发生饱和的格式也不能作为通用硬件规范。</p></article>
    </section>

    <h2>3. camera 0 选出的格式，为什么到其他相机会失败</h2>
    <table><thead><tr><th>信号</th><th>camera 0 最小格式</th><th>camera 0 安全格式</th>
    <th>camera 0 范围</th><th>37 张 held-out 总范围</th></tr></thead><tbody>
    {''.join(format_rows)}</tbody></table>
    <p>最直观的两个例子：mean_x 的负方向从约 −1837 扩到 −6663；RGB 最大值从 3.39 增到 4.55。
    所以 <code>UQ2.8&lt;10&gt;</code> 虽然在 camera 0 装得下 RGB，在其他相机上却会把大于约 4 的值硬夹住。</p>

    <h2>4. 联合量化：位宽、画质和稳定性的真实取舍</h2>
    <table><thead><tr><th>联合 profile</th><th>平均 ΔPSNR@GT</th><th>最差 ΔPSNR@GT</th>
    <th>平均 direct PSNR</th><th>最差 direct PSNR</th><th>饱和次数</th></tr></thead><tbody>
    {''.join(profile_rows)}</tbody></table>
    <p><code>diagnostic_balanced</code> 在 T&T 的 37 张 held-out 相机上零饱和，平均 ΔPSNR 为 −0.0033 dB，
    最差为 −0.0721 dB，最差 direct PSNR 为 54.24 dB。它是很有希望的工程候选。
    但这三组 diagnostic profile 是在看过 T&T held-out 失败原因后设计的，因此这些相机不能再被当作完全独立的最终验证集。</p>

    <h2>5. 各输入单独量化时发生了什么</h2>
    <table><thead><tr><th>信号</th><th>camera 0 选择</th><th>held-out 平均 ΔPSNR</th>
    <th>held-out 最差 ΔPSNR</th><th>最差 direct PSNR</th><th>饱和</th></tr></thead><tbody>
    {''.join(single_rows)}</tbody></table>
    <p>conic 的难点不只是动态范围，而是小数精度：它乘上 dx²、dy² 后，微小误差会改变 power，
    随后又经过 EXP、alpha 阈值和 T-stop，被非线性放大。x/y 和 RGB 则同时面临“范围装不下”和“精度不够”两类问题。</p>

    <h2>6. 图像看起来几乎一样，为什么仍要看数字</h2>
    <section class="grid g3 shots">
      <article class="card"><h3>标准 renderCUDA</h3><img src="{standard}" alt="标准图像"></article>
      <article class="card"><h3>camera 0：mean_x 最小格式</h3><img src="{mean_image}" alt="mean x 量化图像"></article>
      <article class="card"><h3>camera 0：conic_C 最小格式</h3><img src="{conic_image}" alt="conic c 量化图像"></article>
    </section>
    <p>肉眼很难在整图缩略图里发现少量像素的 T-stop 漂移，所以报告同时保留最大像素误差、direct PSNR、
    contributor/T-stop 变化和饱和计数。图片负责直观，数字负责门禁。</p>

    <h2>7. 这套社区框架下一步如何完整闭环</h2>
    <section class="grid g2">
      <article class="card good"><h3>已经完成</h3><ul>
      <li>标准 renderCUDA 契约检查；</li><li>七个输入的独立 Q 格式和真实整图扫描；</li>
      <li>camera 0 校准、37 相机 held-out 复查；</li><li>联合量化、高精度收敛控制组和可追溯 CSV/JSON。</li></ul></article>
      <article class="card warn"><h3>下一阶段</h3><ol>
      <li>用八场景 camera 0 组成真正的跨场景校准集；</li>
      <li>冻结输入格式后，单独扫描二次型中间乘积与求和保护位；</li>
      <li>再独立扫描 EXP 地址 8/9/10 bit 与输出位宽；</li>
      <li>最后扫描 alpha、T 和 RGB accumulator，避免误差来源混在一起。</li></ol></article>
    </section>
    <div class="callout"><b>证据边界：</b>这些结果来自真实 CUDA 整图渲染，能回答画质和数值范围；
    尚不能回答真实面积、频率或功耗。只有格式跨场景冻结、RTL bit-exact 通过并完成相同约束综合后，才能给 PPA 结论。</div>

    <h2>附录：可追溯文件</h2>
    <p><code>camera0_input_sweep.csv</code>：camera 0 单因素扫描；
    <code>heldout_input_validation.csv</code>：原始 held-out 结果；
    <code>controls_v2/heldout_input_validation.csv</code>：宽度诊断和收敛控制；
    <code>profile_summary.csv</code>：本页联合 profile 汇总；
    <code>summary.json</code>：机器可读结论。</p>
    </main></body></html>"""
    output.write_text(body, encoding="utf-8")
    summary["report_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (results / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary

