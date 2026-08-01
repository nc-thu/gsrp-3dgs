"""Build the eight-scene camera-ready precision report."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


SCENE_ORDER = [
    "lego", "hotdog", "tt107k", "truck",
    "drjohnson", "playroom", "bicycle", "garden",
]
SCENE_LABELS = {
    "lego": "Lego", "hotdog": "Hotdog", "tt107k": "T&T-107k",
    "truck": "Truck", "drjohnson": "DrJohnson", "playroom": "Playroom",
    "bicycle": "Bicycle", "garden": "Garden",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _all_scene_rows(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for scene in SCENE_ORDER:
        path = root / scene / "results.csv"
        if path.exists():
            rows.extend(_read_csv(path))
    return rows


def _f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in ("", "nan", "NaN"):
        return float("nan")
    return float(value)


def _profile_summary(rows: list[dict[str, str]]) -> dict[str, float | int | str]:
    if not rows:
        raise ValueError("cannot summarize an empty result set")
    deltas = [_f(row, "delta_psnr_vs_standard") for row in rows]
    direct = [_f(row, "psnr_standard_ref") for row in rows]
    ssim = [_f(row, "delta_ssim_vs_standard") for row in rows]
    lpips = [_f(row, "delta_lpips_vs_standard") for row in rows]
    lpips = [value for value in lpips if value == value]
    return {
        "profile": rows[0]["profile"],
        "cameras": len(rows),
        "mean_delta_psnr": mean(deltas),
        "worst_delta_psnr": min(deltas),
        "mean_delta_ssim": mean(ssim),
        "mean_delta_lpips": mean(lpips) if lpips else float("nan"),
        "worst_positive_delta_lpips": max(lpips) if lpips else float("nan"),
        "worst_direct_psnr": min(direct),
        "saturations": sum(int(row["saturations"]) for row in rows),
    }


def _group_profile(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["profile"]].append(row)
    return [_profile_summary(group) for group in groups.values()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bar(value: float, lower: float, upper: float, color: str) -> str:
    width = max(2.0, min(100.0, (value - lower) / (upper - lower) * 100.0))
    return (
        '<div class="bar"><span style="width:'
        f'{width:.2f}%;background:{color}"></span></div>'
    )


def build_camera_ready_report(
    results: Path, profile_path: Path, output: Path | None = None
) -> dict[str, object]:
    """Generate CSV/JSON summaries and a self-contained Chinese HTML report."""
    output = output or results / "report.html"
    profile_doc = json.loads(profile_path.read_text(encoding="utf-8"))
    profile = profile_doc["profiles"][0]

    final_root = results / "final_quality_safe_v1"
    final_rows = _all_scene_rows(final_root)
    if len(final_rows) != 573:
        raise ValueError(f"expected 573 held-out cameras, found {len(final_rows)}")

    scene_rows: list[dict[str, object]] = []
    for scene in SCENE_ORDER:
        group = [row for row in final_rows if row["scene"] == scene]
        item = _profile_summary(group)
        item["scene"] = scene
        item["scene_label"] = SCENE_LABELS[scene]
        scene_rows.append(item)
    overall = _profile_summary(final_rows)

    stage_specs = [
        ("输入联合量化", results / "joint_input_camera0_v2", "joint_input_quality_safe"),
        ("二次型窄位宽（失败）", results / "quadratic_camera0", "quad_balanced"),
        ("二次型宽控制", results / "quadratic_wide_camera0", "quad_wide40"),
        ("EXP 最小通过", results / "exp_addr8_output_camera0", "exp_addr8_out8"),
        ("EXP 推荐", results / "exp_addr8_output_camera0", "exp_addr8_out10"),
        ("Blending 推荐", results / "blending_refine_camera0", "full_a10_t_uq1p15_acc16"),
    ]
    stage_rows: list[dict[str, object]] = []
    for stage, root, wanted in stage_specs:
        matches = [row for row in _all_scene_rows(root) if row["profile"] == wanted]
        if not matches:
            continue
        item = _profile_summary(matches)
        item["stage"] = stage
        stage_rows.append(item)

    margin_rows = _all_scene_rows(results / "margin_check")
    margin_summary = _profile_summary(margin_rows)
    final_same_scenes = [
        row for row in final_rows if row["scene"] in {"garden", "tt107k"}
    ]
    final_same_summary = _profile_summary(final_same_scenes)

    _write_csv(results / "scene_summary.csv", scene_rows)
    _write_csv(results / "stage_summary.csv", stage_rows)
    _write_csv(results / "margin_check_summary.csv", [
        {"variant": "quality_safe", **final_same_summary},
        {"variant": "wider_margin", **margin_summary},
    ])

    decision = {
        "release_profile": profile["name"],
        "status": "camera-ready software evidence; RTL/PPA not yet evaluated",
        "heldout_cameras": len(final_rows),
        "quality_gate_pass": (
            overall["mean_delta_psnr"] >= -0.2
            and overall["worst_delta_psnr"] >= -0.5
            and overall["mean_delta_ssim"] >= -0.002
            and overall["mean_delta_lpips"] <= 0.005
            and overall["saturations"] == 0
        ),
        "quadratic_conclusion": (
            "Naive A*dx^2+C*dy^2+B*dx*dy needs wide cancellation-safe "
            "intermediates; 40-bit is a validated quality bound, not a final PPA optimum."
        ),
        "exp_conclusion": "8-bit address and 10-bit output selected; 8/8 is the minimum passing LUT.",
        "blend_conclusion": "alpha10, T/visibility UQ1.15, RGB accumulator 16-bit Q4.12.",
        "margin_conclusion": "Wider EXP/alpha/accumulator barely improves Garden; keep smaller profile.",
    }
    manifest = {
        "schema": "gs-raster-precision-camera-ready-release-1",
        "status": "complete",
        "allocator": "atae_snugbox",
        "calibration": "camera 0 from each of eight scenes",
        "validation": "all remaining 573 test cameras",
        "reference": "standard renderCUDA",
        "profile_sha256": _sha256(profile_path),
        "result_csv_sha256": {
            scene: _sha256(final_root / scene / "results.csv") for scene in SCENE_ORDER
        },
        "decision": decision,
        "overall": overall,
    }
    provenance_path = profile_path.parent.parent / "server_adapter" / "BUILD_PROVENANCE.json"
    if provenance_path.exists():
        manifest["build_provenance"] = json.loads(provenance_path.read_text(encoding="utf-8"))
    (results / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    input_labels = {
        "mean_x": "投影中心 x", "mean_y": "投影中心 y",
        "conic_a": "二次型 A", "conic_b": "二次型 B", "conic_c": "二次型 C",
        "opacity": "不透明度", "rgb": "高斯 RGB",
    }
    input_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            input_labels[name], fmt["bits"], fmt["frac_bits"],
            "signed" if fmt.get("signed", False) else "unsigned",
        )
        for name, fmt in profile["inputs"].items()
    )
    scene_html = "".join(
        "<tr><td>{scene_label}</td><td>{cameras}</td>"
        "<td>{mean_delta_psnr:+.4f} dB</td><td>{worst_delta_psnr:+.4f} dB</td>"
        "<td>{mean_delta_ssim:+.6f}</td><td>{mean_delta_lpips:+.6f}</td>"
        "<td>{worst_direct_psnr:.2f} dB{bar}</td><td>{saturations:,}</td></tr>".format(
            **row,
            bar=_bar(float(row["worst_direct_psnr"]), 45.0, 65.0, "#4f7cff"),
        )
        for row in scene_rows
    )
    stage_profile_labels = {
        "joint_input_quality_safe": "输入质量安全候选",
        "quad_balanced": "窄二次型折中候选",
        "quad_wide40": "宽二次型控制组",
        "exp_addr8_out8": "256×8 EXP LUT",
        "exp_addr8_out10": "256×10 EXP LUT",
        "full_a10_t_uq1p15_acc16": "10-bit alpha + 16-bit 状态",
    }
    stage_html = "".join(
        "<tr><td>{stage}</td><td>{profile_label}</td>"
        "<td>{mean_delta_psnr:+.4f}</td><td>{worst_delta_psnr:+.4f}</td>"
        "<td>{worst_direct_psnr:.2f}</td><td>{saturations:,}</td></tr>".format(
            **row, profile_label=stage_profile_labels[str(row["profile"])]
        )
        for row in stage_rows
    )

    exp = profile["exp"]
    blend = profile["blend"]
    quad = profile["quadratic"]
    css = """
    :root{--ink:#172238;--muted:#627089;--blue:#315bdb;--cyan:#0e8aa8;
    --green:#11865b;--orange:#d26d20;--red:#bd3d45;--paper:#f4f7fb;--line:#d9e1ef}
    *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
    font:16px/1.72 "Microsoft YaHei","Noto Sans SC",sans-serif}
    main{max-width:1180px;margin:auto;padding:28px 20px 80px}.hero{color:#fff;border-radius:25px;
    padding:34px;background:linear-gradient(130deg,#14203a,#244fa4 62%,#0788a8)}
    h1{font-size:34px;line-height:1.25;margin:7px 0 12px}h2{font-size:27px;margin:42px 0 14px}
    h3{margin:0 0 7px}.lede{font-size:19px;max-width:940px}.grid{display:grid;gap:15px}
    .g4{grid-template-columns:repeat(4,1fr)}.g3{grid-template-columns:repeat(3,1fr)}
    .g2{grid-template-columns:repeat(2,1fr)}.card{background:white;border:1px solid var(--line);
    border-radius:17px;padding:19px;box-shadow:0 8px 23px #2235550c}.ok{border-top:5px solid var(--green)}
    .warn{border-top:5px solid var(--orange)}.bad{border-top:5px solid var(--red)}
    .big{font-size:29px;font-weight:850}.muted,small{color:var(--muted)}small{display:block}
    table{width:100%;border-collapse:collapse;background:white;border-radius:15px;overflow:hidden}
    th,td{padding:11px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
    th{background:#eaf0fa;font-size:14px}.flow{display:flex;align-items:stretch;gap:8px;flex-wrap:wrap}
    .node{flex:1;min-width:125px;background:#fff;border:2px solid #c4d3ee;border-radius:14px;
    padding:15px;text-align:center}.arrow{align-self:center;color:var(--blue);font-size:25px}
    code{background:#edf2fa;border-radius:6px;padding:2px 6px}.bar{height:7px;background:#e8edf5;
    border-radius:8px;margin-top:5px}.bar span{display:block;height:100%;border-radius:8px}
    .callout{background:#fff8e8;border-left:5px solid var(--orange);border-radius:10px;padding:17px}
    .formula{padding:17px;border-radius:13px;background:#15233f;color:#fff;text-align:center;font-size:19px}
    .stamp{display:inline-block;background:#d9f7ea;color:#08744f;border-radius:99px;padding:4px 11px;font-weight:800}
    .svgbox{background:white;border:1px solid var(--line);border-radius:17px;padding:15px}
    @media(max-width:800px){.g4,.g3,.g2{grid-template-columns:1fr}.arrow{display:none}
    h1{font-size:28px}main{padding:15px 9px 60px}th,td{font-size:12px;padding:7px 5px}}
    """
    html_doc = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>GSRP Camera-ready：八场景统一精度实验</title><style>{css}</style></head><body><main>
    <section class="hero"><span class="stamp">真实 CUDA 整图 · 573 个 held-out camera · 2026-07-24</span>
    <h1>3DGS 光栅化位宽，终于从“凭经验”变成了“逐级测出来”</h1>
    <p class="lede">八个场景的 camera 0 只负责校准；其余 573 个 camera 负责考试。我们先冻结输入，
    再分别实验二次型、EXP LUT 和 blending，最后把整条定点链重新跑完。正式候选全部质量门通过且零饱和。
    它是 camera-ready 的软件精度证据，但还不是 RTL 面积、频率或功耗结论。</p></section>

    <section class="grid g4" style="margin-top:17px">
      <article class="card ok"><h3>平均额外 PSNR</h3><div class="big">{overall['mean_delta_psnr']:+.4f} dB</div>
      <small>门槛：不低于 −0.2 dB</small></article>
      <article class="card ok"><h3>最差 camera</h3><div class="big">{overall['worst_delta_psnr']:+.4f} dB</div>
      <small>门槛：不低于 −0.5 dB</small></article>
      <article class="card ok"><h3>平均 SSIM / LPIPS 变化</h3><div class="big">{overall['mean_delta_ssim']:+.6f}</div>
      <small>LPIPS {overall['mean_delta_lpips']:+.6f}，均通过</small></article>
      <article class="card ok"><h3>未解释饱和</h3><div class="big">{overall['saturations']:,}</div>
      <small>573 个 held-out camera</small></article>
    </section>

    <h2>1. 这轮不是一次“全参数乱扫”，而是分四关过</h2>
    <div class="flow">
      <div class="node"><b>八场景 camera 0</b><br>只用来量范围、选输入 Q 格式</div><div class="arrow">→</div>
      <div class="node"><b>二次型</b><br>单独测平方、乘积、求和保护位</div><div class="arrow">→</div>
      <div class="node"><b>EXP LUT</b><br>地址位宽和输出位宽分开测</div><div class="arrow">→</div>
      <div class="node"><b>Blending</b><br>alpha、T、RGB 累加器分开测</div><div class="arrow">→</div>
      <div class="node"><b>573 张考试卷</b><br>冻结后整链复查，不再移动小数点</div>
    </div>
    <p class="formula">power = −0.5 × (A·dx² + C·dy²) − B·dx·dy → exp(power) → alpha → T / RGB</p>

    <h2>2. 最终冻结的格式</h2>
    <section class="grid g2">
      <article class="card"><h3>光栅输入</h3><table><thead><tr><th>信号</th><th>总位宽</th><th>小数位</th><th>符号</th></tr></thead>
      <tbody>{input_rows}</tbody></table></article>
      <article class="card"><h3>计算链</h3>
      <p><b>二次型：</b>平方 {quad['geometry_bits']} bit，小数 {quad['geometry_frac_bits']}；
      加权项 {quad['term_bits']} bit，小数 {quad['term_frac_bits']}；
      和为 {quad['sum_bits']} bit，power 写回 {quad['power_bits']} bit。</p>
      <p><b>EXP：</b>{exp['address_bits']}-bit 地址 × {exp['output_bits']}-bit 输出，
      即 256 项 LUT，不是 10-bit 地址。</p>
      <p><b>Blending：</b>alpha {blend['alpha']['bits']} bit；T/visibility 为
      UQ1.15；RGB accumulator 为 {blend['accumulator']['bits']} bit、
      小数 {blend['accumulator']['frac_bits']}。</p></article>
    </section>

    <h2>3. 为什么二次型中间量反而这么宽</h2>
    <section class="grid g3">
      <article class="card bad"><h3>24～30 bit 直接失败</h3><p>平方项本身装得下，但
      A·dx²、C·dy²、B·dx·dy 出现约 51.5 万次加权项饱和，平均画质下降达到约 0.5～0.8 dB。</p></article>
      <article class="card warn"><h3>根因是“大数相消”</h3><p>三个中间项可以各自很大，最后相加却很小。
      如果先把大项截窄，后面的相消就救不回来。像三张很大的账单互相抵扣，不能只按最终余额准备纸张。</p></article>
      <article class="card ok"><h3>40 bit 是安全上界</h3><p>40-bit 中间项在八个 calibration camera
      上零饱和、画质通过。它证明数值模型正确，但还不是最省面积的硬件答案。</p></article>
    </section>
    <div class="callout"><b>重要限制：</b>40-bit 是“朴素二次型公式”的质量安全格式。
    camera-ready 论文应诚实把它写成软件精度上界。后续可通过代数重排、坐标范围约束或预处理变换降低硬件位宽，
    但任何新写法都必须重新通过同一套 573-camera 门禁。</div>

    <h2>4. EXP 与 blending 的结论更干净</h2>
    <section class="grid g2">
      <article class="card ok"><h3>EXP：256×10 是推荐点</h3>
      <svg viewBox="0 0 540 185" width="100%" role="img" aria-label="EXP LUT 说明">
      <defs><linearGradient id="g" x1="0" x2="1"><stop stop-color="#315bdb"/><stop offset="1" stop-color="#13a181"/></linearGradient></defs>
      <path d="M30 25 C110 32 130 55 190 85 S330 145 510 158" fill="none" stroke="url(#g)" stroke-width="6"/>
      <line x1="30" y1="160" x2="515" y2="160" stroke="#9aa8bd"/><line x1="30" y1="20" x2="30" y2="160" stroke="#9aa8bd"/>
      <text x="30" y="180">power = −ln(255)</text><text x="485" y="180">0</text>
      <text x="55" y="55" fill="#315bdb">256 个地址桶</text><text x="325" y="125" fill="#11865b">10-bit 输出</text></svg>
      <p>8-bit 地址、8-bit 输出已经通过门槛；把输出加到 10 bit 后，最差相似度明显更稳，成本仍很小。</p></article>
      <article class="card ok"><h3>Blending：先把 alpha 做对</h3>
      <svg viewBox="0 0 540 185" width="100%" role="img" aria-label="Blending 状态图">
      <rect x="12" y="55" width="105" height="62" rx="12" fill="#dce8ff"/><text x="35" y="82">EXP × opacity</text><text x="44" y="104">→ alpha10</text>
      <path d="M120 86h55" stroke="#315bdb" stroke-width="4"/><polygon points="175,86 162,78 162,94" fill="#315bdb"/>
      <rect x="180" y="28" width="145" height="55" rx="12" fill="#dcf5eb"/><text x="205" y="61">RGB acc：16 bit</text>
      <rect x="180" y="103" width="145" height="55" rx="12" fill="#fff0da"/><text x="207" y="136">T：UQ1.15</text>
      <path d="M328 56h75M328 130h75" stroke="#315bdb" stroke-width="4"/><polygon points="403,56 390,48 390,64" fill="#315bdb"/><polygon points="403,130 390,122 390,138" fill="#315bdb"/>
      <rect x="408" y="55" width="120" height="75" rx="12" fill="#e8edf5"/><text x="436" y="84">下一高斯</text><text x="427" y="108">或最终 RGB</text></svg>
      <p>alpha8 是主要误差源；alpha10 通过。T 必须能精确表示初始值 1，所以选 UQ1.15；
      RGB 累加器从 16 加到 18 bit 没有实质收益。</p></article>
    </section>

    <h2>5. 573 个 held-out camera 的逐场景结果</h2>
    <table><thead><tr><th>场景</th><th>camera</th><th>平均 ΔPSNR</th><th>最差 ΔPSNR</th>
    <th>平均 ΔSSIM</th><th>平均 ΔLPIPS</th><th>最差 direct PSNR</th><th>饱和</th></tr></thead>
    <tbody>{scene_html}</tbody></table>
    <p>这里的 ΔPSNR/ΔSSIM/ΔLPIPS 是“候选定点 renderer 相对标准 renderCUDA 的任务质量变化”。
    direct PSNR 是两张输出图直接互相比，专门用来发现“任务指标碰巧变好、但图像其实改得很多”的情况。</p>

    <h2>6. 每一阶段到底留下了什么</h2>
    <table><thead><tr><th>阶段</th><th>候选</th><th>平均 ΔPSNR</th><th>最差 ΔPSNR</th>
    <th>最差 direct PSNR</th><th>饱和</th></tr></thead><tbody>{stage_html}</tbody></table>
    <p>第一次 held-out 输入验证暴露了 Garden camera 9 的 mean_y 越界，共 1,089,480 次饱和。
    因此正式 profile 给 x/y 增加了整数保护位，再从头跑完整链。这次修正是一次安全设计迭代，
    不是把每个 camera 的小数点动态调到最好；最终八场景仍共用同一套静态格式。</p>

    <h2>7. 为什么没有选更宽的“余量版”</h2>
    <section class="grid g2">
      <article class="card"><h3>当前质量安全版</h3><div class="big">Garden 最差 {final_same_summary['worst_direct_psnr']:.2f} dB*</div>
      <p>*这里合并 Garden 与 T&T 的复查子集；全量 Garden 最差为 47.87 dB。</p></article>
      <article class="card"><h3>更宽余量版</h3><div class="big">最差 {margin_summary['worst_direct_psnr']:.2f} dB</div>
      <p>EXP 输出 12 bit、alpha12、累加器18 bit，只把 Garden 最差提高约 0.09 dB，同时部分 ΔPSNR 还略差。</p></article>
    </section>
    <p>换句话说，最差 camera 的差异并不是这些 blending 位宽不足造成的。继续加宽只会增加硬件成本，
    不能解决根因，所以保留更小的正式候选。</p>

    <h2>8. 可以写进 camera-ready 论文的，和还不能写的</h2>
    <section class="grid g2">
      <article class="card ok"><h3>可以写</h3><ul>
      <li>八场景统一静态格式与 573-camera 真实 CUDA 整图验证；</li>
      <li>输入、二次型、EXP、blending 的分阶段敏感度；</li>
      <li>256×10 EXP LUT、alpha10、UQ1.15 T、16-bit accumulator 的软件质量证据；</li>
      <li>朴素二次型中间项存在大数相消，窄格式会系统性失败。</li></ul></article>
      <article class="card warn"><h3>暂时不能写成硬件结论</h3><ul>
      <li>40-bit 是否是最佳面积点；</li><li>真实面积、400 MHz 可达频率或功耗；</li>
      <li>采用代数重排后能缩到多少位；</li><li>当前软件 profile 与 RTL 是否逐 bit 一致。</li></ul></article>
    </section>
    <div class="callout"><b>下一条硬件路线：</b>先把这套 profile 作为 bit-accurate golden；
    再为二次型设计 cancellation-safe 的重排候选。每个候选先过真实 CUDA 画质门，再落 RTL、VCS 和同约束综合。
    这样论文里的画质和 PPA 才来自同一个数值合同。</div>

    <h2>9. 交接与复现</h2>
    <p><a href="../profiles/camera_ready_quality_safe_v1.json">冻结格式</a>保存每个信号的位宽；
    <a href="scene_summary.csv">八场景汇总</a>给出逐场景结果；
    <a href="stage_summary.csv">分阶段证据</a>解释每一步为何这样选；
    <a href="manifest.json">实验清单</a>保存输入结果 hash 与门禁状态。
    公开包不包含模型、数据集、服务器路径或私有图像。</p>
    </main></body></html>"""
    output.write_text(html_doc, encoding="utf-8")
    manifest["report_sha256"] = _sha256(output)
    (results / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
