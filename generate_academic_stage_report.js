const fs = require("fs");
const path = require("path");
const {
  Document,
  Packer,
  Paragraph,
  TextRun,
  HeadingLevel,
  AlignmentType,
  Table,
  TableRow,
  TableCell,
  WidthType,
  BorderStyle,
  ShadingType,
  LevelFormat,
  Footer,
  PageNumber,
} = require("docx");
const PptxGenJS = require("pptxgenjs");

const desktop = path.join(process.env.USERPROFILE || "C:\\Users\\oc200", "Desktop");
const outDir = path.join(desktop, "SGDA_DEAP阶段报告");
fs.mkdirSync(outDir, { recursive: true });

const reportPath = path.join(outDir, "SGDA_DEAP跨被试情感识别阶段报告.docx");
const pptPath = path.join(outDir, "SGDA_DEAP跨被试情感识别阶段汇报.pptx");

const C = {
  navy: "0F172A",
  teal: "0F766E",
  tealLight: "CCFBF1",
  gray: "475569",
  pale: "F8FAFC",
  line: "CBD5E1",
  amber: "B45309",
  red: "B91C1C",
  green: "15803D",
  white: "FFFFFF",
};

const experiments = [
  ["原有 SGDA 基线", "DEAP valence, bd128, bs64", "65.81±7.23%", "59.87±11.12%", "原始对照方法"],
  ["Raw/DE→Riemann 表征", "Tangent, adaptive, bd128, bs8", "65.39±7.70%", "59.23±12.15%", "接近基线，但 batch size 不一致"],
  ["DE→SPD/Riemann 融合", "adaptive, bd128, bs64", "62.15±7.16%", "54.76±8.27%", "直接融合暂未优于基线"],
  ["SPD Align 尝试", "adaptive, bd128, bs64", "59.68±7.89%", "49.13±8.41%", "几何对齐收益有限"],
  ["Tangent 辅助正则", "lambda=0.001, bd128, bs64", "60.68±8.25%", "49.25±11.60%", "辅助约束尚未稳定提升"],
  ["Tangent 辅助正则调试", "bd128, bs16，仅 3 个目标被试", "100.00±0.00%", "100.00±0.00%", "调试结果，不作为正式结论"],
];

function t(text, opts = {}) {
  return new TextRun({
    text,
    font: "Microsoft YaHei",
    size: opts.size || 22,
    bold: !!opts.bold,
    italics: !!opts.italics,
    color: opts.color || C.navy,
  });
}

function p(text, opts = {}) {
  return new Paragraph({
    children: [t(text, opts)],
    heading: opts.heading,
    alignment: opts.align,
    spacing: { before: opts.before ?? 80, after: opts.after ?? 80, line: 320 },
  });
}

function bullet(text) {
  return new Paragraph({
    children: [t(text)],
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 35, after: 35, line: 300 },
  });
}

function cell(text, width, header = false) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: header ? { fill: C.tealLight, type: ShadingType.CLEAR } : undefined,
    margins: { top: 90, bottom: 90, left: 110, right: 110 },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 1, color: C.line },
      bottom: { style: BorderStyle.SINGLE, size: 1, color: C.line },
      left: { style: BorderStyle.SINGLE, size: 1, color: C.line },
      right: { style: BorderStyle.SINGLE, size: 1, color: C.line },
    },
    children: [new Paragraph({
      children: [t(text, { size: header ? 19 : 18, bold: header })],
      spacing: { before: 0, after: 0 },
    })],
  });
}

function table(rows, widths) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((row, i) => new TableRow({
      children: row.map((v, j) => cell(v, widths[j], i === 0)),
    })),
  });
}

async function makeDocx() {
  const body = [];

  body.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 260, after: 120 },
    children: [t("基于 SGDA 的 DEAP 跨被试情感识别阶段报告", { size: 34, bold: true, color: C.teal })],
  }));
  body.push(p("研究主题：Raw EEG→SPD/Riemann 几何表征在 SGDA 基线上的探索 | 2026-06-05", {
    align: AlignmentType.CENTER,
    color: C.gray,
    after: 260,
  }));

  body.push(p("一、研究背景与阶段定位", { heading: HeadingLevel.HEADING_1 }));
  body.push(p("本阶段工作并非重新提出 SGDA。SGDA 是代码库中已有的跨被试 EEG 情感识别基线方法，核心思想是利用共享特征提取器、多源域特定分支、语义原型对齐与域分布约束提升跨被试泛化能力。本阶段的研究重点是在原有 SGDA 基线基础上，面向 DEAP 数据集探索 Riemannian geometry 表征是否能够进一步缓解不同被试之间的分布差异。"));
  body.push(p("具体而言，阶段性尝试包括：从 Raw EEG 或 DE 特征构造 SPD 协方差矩阵；将 SPD 矩阵映射到 Riemann 切空间；比较几何特征作为输入、几何特征与 SGDA 融合、以及几何辅助正则对原有 SGDA 表征的影响。"));

  body.push(p("二、原有 SGDA 基线理解", { heading: HeadingLevel.HEADING_1 }));
  [
    "输入形式：原 SGDA 主要使用 DE 特征，样本形状可表示为 [T, C, F]，其中 T 为时间片，C 为通道数，F 为频带数。",
    "共享特征提取：SFE 对 EEG 特征进行统一编码，得到跨域共享表征。",
    "多源域建模：每一个源被试对应一个 DSFE 分支，以保留不同源域的个体差异。",
    "语义原型对齐：情感标签文本 positive/negative 被编码为语义向量，模型通过 align_loss 将 EEG 表征对齐到对应文本原型。",
    "跨域约束：MMD 用于缩小源域与目标域分布差异，discrepancy_loss 用于约束多个源分支在目标域上的输出一致性。",
    "评估融合：根据目标样本到源域 centroid 的距离进行 sample-wise 加权融合，得到最终预测。",
  ].forEach((x) => body.push(bullet(x)));

  body.push(p("三、本人阶段性修改与尝试", { heading: HeadingLevel.HEADING_1 }));
  body.push(table([
    ["方向", "实现思路", "研究目的"],
    ["Raw EEG→SPD", "将原始 EEG 片段视为多通道时间序列，估计通道协方差矩阵，并通过正则化保证 SPD 性质。", "显式建模通道间协方差结构，补充 DE 特征的统计信息。"],
    ["SPD→Tangent", "利用 Riemannian geometry 将 SPD 矩阵映射到切空间，得到欧氏空间可训练向量。", "将非欧几里得结构转换为可与神经网络结合的特征形式。"],
    ["几何输入模型", "将切空间向量作为模型输入，替代或补充原有 DE 表征。", "检验几何表征本身是否具有判别能力。"],
    ["几何融合/对齐", "比较 adaptive fusion、average fusion、SPD align 等策略。", "探索几何域对齐能否改善跨被试迁移。"],
    ["Tangent 辅助正则", "保留原 SGDA 主推理路径，仅在训练时用切空间投影约束 SFE 输出。", "在不增加推理复杂度的前提下引入几何先验。"],
  ], [1500, 4500, 3360]));

  body.push(p("四、Raw→SPD→Tangent 技术路线", { heading: HeadingLevel.HEADING_1 }));
  body.push(p("Raw→SPD 的关键是将 EEG 样本表示为通道维度上的协方差矩阵。对于每个样本 X∈R^{C×T}，可估计协方差矩阵 Σ，并加入 εI 保证数值稳定性。由于 SPD 矩阵位于非欧几里得流形，直接输入普通神经网络可能破坏其几何结构，因此进一步通过 Riemannian tangent space 将其映射为向量表示。该向量可作为模型输入，也可作为辅助监督信号约束 SGDA 中的 SFE 表征。"));
  [
    "Raw EEG 路径：Raw EEG → covariance estimation → SPD matrix → tangent vector → classifier/SGDA branch。",
    "DE 几何路径：DE feature → reshape to channel-wise representation → SPD matrix → tangent vector。",
    "辅助正则路径：DE 主分支保持不变；tangent vector 经线性投影后，与 SFE 输出计算 MSE。",
    "推理复杂度控制：Tangent Aux 方案在推理阶段仍只使用 SGDA 主路径，几何分支仅用于训练阶段。",
  ].forEach((x) => body.push(bullet(x)));

  body.push(p("五、实验结果", { heading: HeadingLevel.HEADING_1 }));
  body.push(table([
    ["方法", "实验设置", "Acc", "Macro-F1", "备注"],
    ...experiments,
  ], [1700, 2500, 1250, 1450, 2460]));
  body.push(p("从当前结果看，完整 32 被试 DEAP valence 设置下，原有 SGDA 基线仍取得最高准确率 65.81%。Raw/DE→Riemann 表征在部分设置下达到 65.39%，说明 SPD/Riemann 几何特征具有一定潜力；但在 batch size 与评估协议未完全一致的情况下，不能直接判定其超过基线。SPD Align、DE→SPD/Riemann 融合和 Tangent 辅助正则在完整设置下均低于 SGDA 基线，提示当前几何信息引入方式仍需进一步优化。"));
  body.push(p("特别说明：Tangent 辅助正则中 bs16 的 100% 结果仅覆盖 3 个目标被试，属于调试性结果，不能作为正式实验结论。"));

  body.push(p("六、阶段性分析", { heading: HeadingLevel.HEADING_1 }));
  [
    "几何特征可能有效，但当前融合方式尚未充分解决语义判别目标与几何结构目标之间的冲突。",
    "若 TangentSpace 按被试独立拟合，切空间坐标系可能不一致，从而削弱跨被试对齐效果。",
    "Raw→SPD 强调通道协方差结构，DE 特征强调频带能量统计，两者表征对象不同，简单拼接或 MSE 约束可能不足以产生互补收益。",
    "不同实验之间 batch size、训练轮数、目标被试覆盖范围尚需完全统一，否则阶段比较容易产生偏差。",
    "下一步应将 SGDA 作为固定对照，系统开展几何表征路径、损失权重、融合策略和评估协议的消融实验。",
  ].forEach((x) => body.push(bullet(x)));

  body.push(p("七、下一阶段计划", { heading: HeadingLevel.HEADING_1 }));
  body.push(table([
    ["优先级", "计划内容", "预期结果"],
    ["高", "统一 DEAP 实验协议：32 被试、相同 batch size、相同 epoch、相同随机种子。", "获得可公平比较的基线与修改方案结果。"],
    ["高", "区分 Raw→SPD、DE→SPD、SPD→Tangent 三条路径，分别单独评估。", "明确几何信息真正有效的入口。"],
    ["高", "进行 λ 扫描与 loss 量级记录，比较 λ=0、1e-5、1e-4、1e-3、1e-2。", "判断辅助几何约束是否过强或无效。"],
    ["中", "尝试共享 TangentSpace 参考点或基于源域统一拟合切空间。", "减少跨被试切空间坐标不一致问题。"],
    ["中", "开展 fusion 消融：average、adaptive、feature-level、decision-level。", "定位融合策略对结果的影响。"],
  ], [900, 5400, 3060]));

  const doc = new Document({
    styles: {
      default: { document: { run: { font: "Microsoft YaHei", size: 22 } } },
      paragraphStyles: [
        {
          id: "Heading1",
          name: "Heading 1",
          basedOn: "Normal",
          next: "Normal",
          quickFormat: true,
          run: { font: "Microsoft YaHei", size: 29, bold: true, color: C.teal },
          paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 },
        },
      ],
    },
    numbering: {
      config: [{
        reference: "bullets",
        levels: [{
          level: 0,
          format: LevelFormat.BULLET,
          text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 520, hanging: 260 } } },
        }],
      }],
    },
    sections: [{
      properties: {
        page: {
          size: { width: 11906, height: 16838 },
          margin: { top: 1080, right: 900, bottom: 1080, left: 900 },
        },
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [t("第 ", { size: 18, color: C.gray }), PageNumber.CURRENT, t(" 页", { size: 18, color: C.gray })],
          })],
        }),
      },
      children: body,
    }],
  });

  fs.writeFileSync(reportPath, await Packer.toBuffer(doc));
}

function slideTitle(slide, title, subtitle) {
  slide.addText(title, { x: 0.5, y: 0.32, w: 12.1, h: 0.45, fontFace: "Microsoft YaHei", fontSize: 24, bold: true, color: C.navy, margin: 0 });
  if (subtitle) slide.addText(subtitle, { x: 0.52, y: 0.82, w: 11.2, h: 0.24, fontFace: "Microsoft YaHei", fontSize: 9.5, color: C.gray, margin: 0 });
}

function footer(slide, n) {
  slide.addText(`SGDA-DEAP 阶段汇报 | ${n}`, { x: 11.15, y: 7.12, w: 1.55, h: 0.16, fontFace: "Microsoft YaHei", fontSize: 7, color: "64748B", margin: 0 });
}

function card(slide, x, y, w, h, title, body, accent = C.teal) {
  slide.addShape("rect", { x, y, w, h, fill: { color: C.white }, line: { color: "E2E8F0", width: 0.8 }, shadow: { type: "outer", color: "000000", opacity: 0.08, blur: 3, angle: 45, distance: 1 } });
  slide.addShape("rect", { x, y, w: 0.08, h, fill: { color: accent }, line: { color: accent } });
  slide.addText(title, { x: x + 0.18, y: y + 0.15, w: w - 0.32, h: 0.24, fontFace: "Microsoft YaHei", fontSize: 12, bold: true, color: C.navy, margin: 0 });
  slide.addText(body, { x: x + 0.18, y: y + 0.52, w: w - 0.32, h: h - 0.6, fontFace: "Microsoft YaHei", fontSize: 9.3, color: C.gray, margin: 0.02, fit: "shrink" });
}

function bullets(slide, items, x, y, w, h, size = 11) {
  slide.addText(items.map((text, i) => ({ text, options: { bullet: true, breakLine: i < items.length - 1 } })), {
    x, y, w, h, fontFace: "Microsoft YaHei", fontSize: size, color: C.navy, margin: 0.03, fit: "shrink",
  });
}

async function makePptx() {
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "Codex";
  pptx.subject = "SGDA DEAP stage report";
  pptx.title = "基于 SGDA 的 DEAP 跨被试情感识别阶段汇报";
  pptx.lang = "zh-CN";
  pptx.theme = { headFontFace: "Microsoft YaHei", bodyFontFace: "Microsoft YaHei", lang: "zh-CN" };

  let s = pptx.addSlide();
  s.background = { color: C.navy };
  s.addShape("rect", { x: 0.68, y: 0.75, w: 0.12, h: 5.55, fill: { color: C.teal }, line: { color: C.teal } });
  s.addText("基于 SGDA 的 DEAP 跨被试情感识别", { x: 1.05, y: 1.65, w: 10.8, h: 0.55, fontFace: "Microsoft YaHei", fontSize: 30, bold: true, color: C.white, margin: 0 });
  s.addText("阶段汇报：Raw EEG→SPD/Riemann 几何表征探索", { x: 1.08, y: 2.45, w: 9.8, h: 0.32, fontFace: "Microsoft YaHei", fontSize: 16, color: C.tealLight, margin: 0 });
  s.addText("说明：SGDA 为原有基线，本阶段工作是在 DEAP 数据集上围绕 SPD/Riemann 表征开展尝试。", { x: 1.08, y: 3.15, w: 10.4, h: 0.3, fontFace: "Microsoft YaHei", fontSize: 10.5, color: "CBD5E1", margin: 0 });
  s.addText("2026-06-05", { x: 1.08, y: 6.35, w: 2.0, h: 0.2, fontFace: "Microsoft YaHei", fontSize: 10, color: "94A3B8", margin: 0 });

  s = pptx.addSlide();
  slideTitle(s, "阶段定位", "不是重新提出 SGDA，而是在原有 SGDA 基线上探索几何表征");
  card(s, 0.65, 1.35, 3.65, 1.45, "原有方法", "SGDA 已存在于代码库中，作为 DEAP 跨被试 valence 二分类基线。");
  card(s, 4.85, 1.35, 3.65, 1.45, "本人工作", "围绕 Raw/DE→SPD→Tangent 路径，尝试几何输入、融合和辅助正则。", C.amber);
  card(s, 9.05, 1.35, 3.35, 1.45, "阶段目标", "分析几何结构信息能否改善跨被试泛化。", C.green);
  bullets(s, [
    "SGDA 基线完整 32 被试 Acc 为 65.81%。",
    "Riemann 表征在部分设置下接近基线，但整体尚未稳定超过基线。",
    "当前重点是建立公平协议并定位几何信息的有效入口。",
  ], 0.9, 4.0, 11.2, 1.05, 12);
  footer(s, 2);

  s = pptx.addSlide();
  slideTitle(s, "技术路线：Raw→SPD→Tangent", "将 EEG 通道协方差结构引入 SGDA 框架");
  const nodes = [
    ["Raw EEG", "C×T signal"],
    ["SPD", "covariance + εI"],
    ["Tangent", "Riemann log-map"],
    ["SGDA", "input/fusion/aux"],
    ["Prediction", "weighted fusion"],
  ];
  nodes.forEach((n, i) => {
    const x = 0.8 + i * 2.45;
    s.addShape("roundRect", { x, y: 2.0, w: 1.65, h: 0.9, rectRadius: 0.06, fill: { color: i === 3 ? C.tealLight : C.white }, line: { color: "99F6E4" } });
    s.addText(n[0], { x: x + 0.12, y: 2.22, w: 1.41, h: 0.18, fontFace: "Microsoft YaHei", fontSize: 11, bold: true, color: C.teal, align: "center", margin: 0 });
    s.addText(n[1], { x: x + 0.12, y: 2.53, w: 1.41, h: 0.14, fontFace: "Consolas", fontSize: 7, color: C.gray, align: "center", margin: 0 });
    if (i < nodes.length - 1) s.addShape("line", { x: x + 1.72, y: 2.45, w: 0.55, h: 0, line: { color: C.teal, width: 1.8, endArrowType: "triangle" } });
  });
  bullets(s, [
    "Raw→SPD 主要捕获通道间协方差结构。",
    "SPD→Tangent 将非欧几里得对象转为可训练向量。",
    "几何特征可作为输入，也可作为辅助约束而不改变推理路径。",
  ], 1.0, 4.35, 10.8, 0.85, 11);
  footer(s, 3);

  s = pptx.addSlide();
  slideTitle(s, "本人尝试内容", "围绕几何表征的输入、融合和正则化展开");
  card(s, 0.75, 1.35, 3.65, 1.35, "几何输入", "使用 SPD/Tangent 特征替代或补充 DE 特征，检验几何表征的判别能力。");
  card(s, 4.85, 1.35, 3.65, 1.35, "几何融合", "比较 adaptive、average、SPD align 等策略。", C.amber);
  card(s, 8.95, 1.35, 3.65, 1.35, "辅助正则", "保留 SGDA 主路径，用 tangent 投影约束 SFE 表征。", C.green);
  bullets(s, [
    "这些尝试均以原有 SGDA 为对照，而不是替代 SGDA 的完整新框架。",
    "当前结果显示几何方向有潜力，但融合方式和损失尺度尚需细化。",
  ], 0.9, 4.2, 11.1, 0.75, 12);
  footer(s, 4);

  s = pptx.addSlide();
  slideTitle(s, "实验结果", "完整 32 被试条件下，原有 SGDA 基线仍为当前最优对照");
  const bars = [
    ["SGDA 基线", 65.81, C.teal],
    ["Raw/DE→Riemann", 65.39, C.green],
    ["DE→SPD/Riemann", 62.15, C.amber],
    ["Tangent Aux", 60.68, C.red],
    ["SPD Align", 59.68, "64748B"],
  ];
  bars.forEach((b, i) => {
    const y = 1.38 + i * 0.72;
    s.addText(b[0], { x: 0.9, y: y + 0.05, w: 2.1, h: 0.18, fontFace: "Microsoft YaHei", fontSize: 10, color: C.navy, margin: 0 });
    s.addShape("rect", { x: 3.15, y, w: 6.6, h: 0.32, fill: { color: "E2E8F0" }, line: { color: "E2E8F0" } });
    s.addShape("rect", { x: 3.15, y, w: 6.6 * (b[1] / 70), h: 0.32, fill: { color: b[2] }, line: { color: b[2] } });
    s.addText(`${b[1].toFixed(2)}%`, { x: 9.95, y: y + 0.05, w: 0.95, h: 0.16, fontFace: "Consolas", fontSize: 9, bold: true, color: C.navy, margin: 0 });
  });
  s.addText("注：Tangent Aux bs16 的 100% 仅覆盖 3 个目标被试，属于调试结果，不作为正式性能结论。", {
    x: 0.9, y: 5.75, w: 11.2, h: 0.24, fontFace: "Microsoft YaHei", fontSize: 9.5, color: C.red, margin: 0,
  });
  footer(s, 5);

  s = pptx.addSlide();
  slideTitle(s, "阶段性分析", "几何特征接近有效，但当前引入方式还不稳定");
  card(s, 0.8, 1.35, 3.65, 1.45, "可能原因一", "不同被试独立拟合 TangentSpace，可能导致切空间坐标系不一致。", C.red);
  card(s, 4.85, 1.35, 3.65, 1.45, "可能原因二", "Raw→SPD 与 DE 表征侧重点不同，简单融合可能产生目标冲突。", C.amber);
  card(s, 8.9, 1.35, 3.55, 1.45, "可能原因三", "辅助损失 λ 虽小，但 MSE 与主损失尺度可能不匹配。", C.teal);
  bullets(s, [
    "下一步需要先统一实验协议，再比较方法优劣。",
    "重点做 λ 扫描、切空间参考点统一、fusion 消融。",
  ], 1.0, 4.3, 10.8, 0.75, 12);
  footer(s, 6);

  s = pptx.addSlide();
  slideTitle(s, "下一阶段计划", "以原 SGDA 为固定基线，系统验证 Raw→SPD/Riemann 路径");
  const plans = [
    ["1", "统一协议", "32 被试、同 batch size、同 epoch、同 seed。"],
    ["2", "路径拆分", "Raw→SPD、DE→SPD、SPD→Tangent 分别评估。"],
    ["3", "损失扫描", "λ=0/1e-5/1e-4/1e-3/1e-2 并记录 loss 量级。"],
    ["4", "坐标统一", "尝试共享 TangentSpace 或源域统一拟合。"],
    ["5", "消融分析", "比较 feature/decision、average/adaptive fusion。"],
  ];
  plans.forEach((p, i) => {
    const y = 1.3 + i * 0.82;
    s.addShape("oval", { x: 0.9, y, w: 0.42, h: 0.42, fill: { color: C.teal }, line: { color: C.teal } });
    s.addText(p[0], { x: 0.9, y: y + 0.1, w: 0.42, h: 0.12, fontFace: "Consolas", fontSize: 9, bold: true, color: C.white, align: "center", margin: 0 });
    s.addText(p[1], { x: 1.55, y: y + 0.05, w: 1.7, h: 0.18, fontFace: "Microsoft YaHei", fontSize: 12, bold: true, color: C.navy, margin: 0 });
    s.addText(p[2], { x: 3.45, y: y + 0.06, w: 8.2, h: 0.16, fontFace: "Microsoft YaHei", fontSize: 10.5, color: C.gray, margin: 0 });
  });
  footer(s, 7);

  s = pptx.addSlide();
  s.background = { color: C.navy };
  s.addText("阶段结论", { x: 0.9, y: 1.0, w: 3.0, h: 0.45, fontFace: "Microsoft YaHei", fontSize: 30, bold: true, color: C.white, margin: 0 });
  s.addText("SGDA 是原有基线。本阶段的主要贡献是在 DEAP 数据集上围绕 Raw EEG→SPD→Riemann/Tangent 几何表征开展探索。当前几何方法尚未稳定超过 SGDA，但 Riemann 表征接近基线，说明该方向具备继续研究价值。", {
    x: 0.95, y: 2.0, w: 11.0, h: 1.1, fontFace: "Microsoft YaHei", fontSize: 18, color: "E2E8F0", margin: 0.03, fit: "shrink",
  });
  s.addText("后续重点：统一协议、拆分几何路径、控制损失尺度、完成系统消融。", {
    x: 0.95, y: 4.25, w: 9.8, h: 0.26, fontFace: "Microsoft YaHei", fontSize: 12, color: C.tealLight, margin: 0,
  });
  footer(s, 8);

  await pptx.writeFile({ fileName: pptPath });
}

(async () => {
  await makeDocx();
  await makePptx();
  console.log(reportPath);
  console.log(pptPath);
})();
