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
  PageNumber,
  Footer,
} = require("docx");
const PptxGenJS = require("pptxgenjs");

const outDir = path.join(__dirname, "阶段报告输出");
fs.mkdirSync(outDir, { recursive: true });

const reportPath = path.join(outDir, "SGDA脑机接口情感识别阶段报告.docx");
const pptPath = path.join(outDir, "SGDA脑机接口情感识别阶段汇报.pptx");

const colors = {
  ink: "1F2937",
  muted: "4B5563",
  teal: "0F766E",
  mint: "CCFBF1",
  pale: "F8FAFC",
  line: "CBD5E1",
  amber: "F59E0B",
  red: "DC2626",
  green: "16A34A",
  navy: "0F172A",
  white: "FFFFFF",
};

const resultRows = [
  ["方法/实验", "设置", "Acc", "Macro-F1", "说明"],
  ["SGDA 基线", "DEAP valence, bd128, bs64", "65.81±7.23%", "59.87±11.12%", "当前主基线，完整 32 被试"],
  ["Riemann 输入模型", "Tangent, adaptive, bd128, bs8", "65.39±7.70%", "59.23±12.15%", "与基线接近，但 batch size 不同"],
  ["DE→Riemann 融合", "adaptive, bd128, bs64", "62.15±7.16%", "54.76±8.27%", "直接融合低于基线"],
  ["SPD Align", "adaptive, bd128, bs64", "59.68±7.89%", "49.13±8.41%", "几何对齐收益不足"],
  ["Tangent Aux", "lambda=0.001, bd128, bs64", "60.68±8.25%", "49.25±11.60%", "完整 32 被试结果低于基线"],
  ["Tangent Aux 调试", "bd128, bs16", "100.00±0.00%", "100.00±0.00%", "仅 3 个目标被试，不能作为最终结论"],
  ["SEED 跨 session", "bd512, bs128", "94.89±4.09%", "94.84±4.13%", "说明框架可迁移到多数据集"],
];

function run(text, opts = {}) {
  return new TextRun({
    text,
    font: "Microsoft YaHei",
    size: opts.size || 22,
    bold: opts.bold || false,
    color: opts.color || colors.ink,
    italics: opts.italics || false,
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    children: [run(text, opts)],
    heading: opts.heading,
    alignment: opts.align,
    spacing: { before: opts.before || 80, after: opts.after || 80, line: 320 },
  });
}

function bullet(text) {
  return new Paragraph({
    children: [run(text)],
    numbering: { reference: "bullet-list", level: 0 },
    spacing: { before: 30, after: 30, line: 300 },
  });
}

function tableCell(text, width, header = false) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: header ? { fill: colors.mint, type: ShadingType.CLEAR } : undefined,
    margins: { top: 100, bottom: 100, left: 120, right: 120 },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 1, color: colors.line },
      bottom: { style: BorderStyle.SINGLE, size: 1, color: colors.line },
      left: { style: BorderStyle.SINGLE, size: 1, color: colors.line },
      right: { style: BorderStyle.SINGLE, size: 1, color: colors.line },
    },
    children: [
      new Paragraph({
        children: [run(text, { bold: header, size: header ? 20 : 18 })],
        spacing: { before: 0, after: 0 },
      }),
    ],
  });
}

function makeTable(rows, widths) {
  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: rows.map((row, i) => new TableRow({
      children: row.map((cell, j) => tableCell(cell, widths[j], i === 0)),
    })),
  });
}

async function buildDocx() {
  const children = [];
  children.push(new Paragraph({
    children: [run("SGDA 脑机接口情感识别阶段报告", { size: 36, bold: true, color: colors.teal })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 300, after: 160 },
  }));
  children.push(para("基于 SGDA_py3.11 工作区代码整理 | 2026-06-05", {
    align: AlignmentType.CENTER,
    color: colors.muted,
    after: 280,
  }));

  children.push(para("一、阶段目标与总体进展", { heading: HeadingLevel.HEADING_1, size: 28, bold: true }));
  children.push(para("本阶段围绕 EEG 情感识别中的跨被试泛化问题展开，重点复现并扩展 SGDA 框架：以差分熵 DE 特征为主输入，通过共享特征提取器、多源域特定分支、文本语义原型对齐和 MMD 域对齐完成跨被试 valence 二分类。代码中已经形成覆盖 DEAP、SEED、SEED-IV、SEED-V、DREAMER 等数据集的加载、预处理、实验脚本和结果记录体系。"));
  [
    "完成 SGDA 基线实验脚本与多源 leave-one-subject-out 训练流程。",
    "引入 CLIP/BERT/SBERT/Roberta 等文本编码接口，将情感标签映射为语义原型。",
    "尝试 Riemann/SPD/切空间几何特征与原 SGDA 结构结合。",
    "新增 Tangent Geometry Auxiliary Regularization：在不改变主推理路径的条件下，用切空间投影约束 SFE 表征。",
    "保存了多组 DEAP 与 SEED 实验结果，具备阶段性对比分析基础。",
  ].forEach((x) => children.push(bullet(x)));

  children.push(para("二、代码结构梳理", { heading: HeadingLevel.HEADING_1, size: 28, bold: true }));
  children.push(makeTable([
    ["模块", "主要文件", "作用"],
    ["配置层", "config/setting.py, utils/args.py", "数据集、切分方式、训练参数、标签边界与会话设置。"],
    ["数据层", "data_utils/load_data.py, preprocess.py, split.py", "读取多数据集 EEG/DE 特征，完成滤波、DE/PSD 提取、LDS 平滑、归一化、分段和划分。"],
    ["语义层", "data_utils/text_to_vector.py, label_text_mapper.py", "把 positive/negative 等类别文本编码为归一化语义原型。"],
    ["模型层", "models/model.py, model_riemann.py", "SGDA 主网络：SFE 共享特征提取器 + DSFE 多源分支 + 语义投影。"],
    ["训练层", "experiments/deap/*.py, experiments/seed*.py", "不同数据集/方法的跨被试、跨 session 实验脚本。"],
    ["损失与评估", "utils/loss.py, mix_utils.py, log_utils.py", "语义对齐、MMD、分支差异、源域加权融合、指标保存。"],
  ], [1500, 2600, 5260]));

  children.push(para("三、核心方法理解", { heading: HeadingLevel.HEADING_1, size: 28, bold: true }));
  children.push(para("SGDA 的主线可以概括为：DE EEG 样本进入共享特征提取器 SFE，得到统一 EEG 表征；每个源被试对应一个 DSFE 分支，将共享表征投影到文本语义空间；源域样本通过 align_loss 向其标签文本原型靠近，源/目标通过 MMD 缩小分布差异，目标样本则通过多个分支的输出差异约束来获得更一致的判别表示。评估阶段按目标样本到各源域 centroid 的距离自适应加权融合，得到最终预测。"));
  children.push(makeTable([
    ["组件", "实现要点", "阶段判断"],
    ["SFE", "Conv2D 处理 [B,T,C,F]，先重排为 [B,1,C*F,T]，再映射到 512 维。", "适合 DEAP/SEED 这类通道×频带结构。"],
    ["DSFE", "每个源域独立 bottleneck 分支，输出 L2 归一化语义向量。", "体现多源域差异，避免一个统一投影吃掉所有个体差异。"],
    ["Semantic Prototype", "CLIP 默认本地模型，标签文本为 negative/positive 等。", "让分类目标从普通 softmax 变为语义空间对齐。"],
    ["MMD + Discrepancy", "MMD 对齐源/目标均值；分支输出差异约束目标预测一致性。", "是跨被试泛化的核心约束。"],
    ["Sample-wise Fusion", "按目标分支表征到源域 centroid 的距离做 softmax 权重。", "比简单平均更符合多源适配直觉。"],
  ], [1500, 3900, 3960]));

  children.push(para("四、几何增强方案", { heading: HeadingLevel.HEADING_1, size: 28, bold: true }));
  children.push(para("新增的 crossSubject_sgda_tangent_aux.py 保留原 SGDA 主分支不变，同时为同一样本从 DE 特征构造 SPD 协方差矩阵并映射到 Riemann 切空间。切空间向量经轻量线性投影器映射到 SFE 维度，用 MSE 约束主网络 SFE 输出，形成 L_geo_aux。该设计的好处是推理阶段仍然只依赖 DE 主分支，不引入额外切空间输入；风险是辅助目标与语义判别目标可能存在梯度冲突，且切空间标准化、lambda、batch size 对结果敏感。"));
  [
    "DE 样本形状：通常为 [T, C, F]，DEAP 中设置 sample_length=3、stride=1。",
    "几何特征路径：DE reshape 为 [C, T*F]，OAS 协方差估计，TangentSpace 映射，subject 内 z-score。",
    "辅助损失：L = L_cls + alpha L_mmd + beta L_disc + 0.001 L_geo_aux。",
    "评估策略：仍使用基线 get_preds，只用 DE 输入，避免把新增分支当作推理捷径。",
  ].forEach((x) => children.push(bullet(x)));

  children.push(para("五、实验结果与阶段结论", { heading: HeadingLevel.HEADING_1, size: 28, bold: true }));
  children.push(makeTable(resultRows, [1750, 2450, 1300, 1450, 2410]));
  children.push(para("从结果看，DEAP valence 完整 32 被试条件下，当前 SGDA 基线 Acc 为 65.81%，仍是最强结果；Riemann 输入模型在 bs8 下达到 65.39%，接近基线但设置不完全一致；Tangent Aux 在完整 32 被试、bs64 下为 60.68%，说明当前辅助几何约束尚未带来稳定收益。bs16 的 100% 结果只覆盖 3 个目标被试，应视为调试结果，不能作为最终性能。"));

  children.push(para("六、问题分析", { heading: HeadingLevel.HEADING_1, size: 28, bold: true }));
  [
    "辅助几何分支可能把 SFE 拉向“协方差结构相似”而非“语义类别可分”的方向，导致主任务性能下降。",
    "TangentSpace 按被试独立拟合，跨被试参考点不一致，可能削弱多源对齐的一致坐标系。",
    "lambda=0.001 虽小，但 MSE 数值尺度可能仍然与主损失不匹配，需要记录各 loss 的量级。",
    "不同实验之间 batch size、目标被试数量不完全一致，阶段比较需要统一协议后再下结论。",
    "README 仍保留 USB 半监督学习模板内容，与当前 EEG/SGDA 项目不匹配，后续应更新项目说明。",
  ].forEach((x) => children.push(bullet(x)));

  children.push(para("七、下一阶段计划", { heading: HeadingLevel.HEADING_1, size: 28, bold: true }));
  children.push(makeTable([
    ["优先级", "任务", "预期产出"],
    ["高", "统一 DEAP 实验协议：epochs、batch size、32 被试、随机种子、评估融合方式。", "可公平比较的主结果表。"],
    ["高", "做 lambda 扫描：0、1e-5、1e-4、1e-3、1e-2，并记录主损失/辅助损失量级。", "判断辅助损失是否过强或无效。"],
    ["中", "尝试共享 TangentSpace 参考点或训练集源域统一拟合，减少坐标系不一致。", "更合理的 Riemann 特征对齐方案。"],
    ["中", "加入消融：无 MMD、无 discrepancy、average fusion、adaptive fusion。", "定位性能贡献来源。"],
    ["中", "整理 README 与实验运行说明。", "便于复现实验和后续交接。"],
  ], [900, 5200, 3260]));

  const doc = new Document({
    styles: {
      default: { document: { run: { font: "Microsoft YaHei", size: 22 } } },
      paragraphStyles: [
        { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 30, bold: true, font: "Microsoft YaHei", color: colors.teal },
          paragraph: { spacing: { before: 260, after: 140 }, outlineLevel: 0 } },
      ],
    },
    numbering: {
      config: [{
        reference: "bullet-list",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 520, hanging: 260 } } } }],
      }],
    },
    sections: [{
      properties: {
        page: { size: { width: 11906, height: 16838 }, margin: { top: 1080, right: 900, bottom: 1080, left: 900 } },
      },
      footers: { default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [run("第 ", { size: 18, color: colors.muted }), PageNumber.CURRENT, run(" 页", { size: 18, color: colors.muted })],
      })] }) },
      children,
    }],
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(reportPath, buffer);
}

function addTitle(slide, title, subtitle) {
  slide.addText(title, { x: 0.45, y: 0.28, w: 9.1, h: 0.46, fontFace: "Microsoft YaHei", fontSize: 24, bold: true, color: colors.navy, margin: 0 });
  if (subtitle) {
    slide.addText(subtitle, { x: 0.48, y: 0.79, w: 8.8, h: 0.24, fontFace: "Microsoft YaHei", fontSize: 8.8, color: colors.muted, margin: 0 });
  }
}

function addFooter(slide, n) {
  slide.addText(`SGDA 阶段汇报 | ${n}`, { x: 8.45, y: 5.26, w: 1.1, h: 0.16, fontFace: "Microsoft YaHei", fontSize: 6.8, color: "64748B", margin: 0 });
}

function addChip(slide, text, x, y, w, fill = colors.mint) {
  slide.addShape("roundRect", { x, y, w, h: 0.33, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
  slide.addText(text, { x: x + 0.07, y: y + 0.08, w: w - 0.14, h: 0.13, fontFace: "Microsoft YaHei", fontSize: 8, bold: true, color: colors.teal, align: "center", margin: 0 });
}

function addCard(slide, x, y, w, h, title, body, accent = colors.teal) {
  slide.addShape("rect", { x, y, w, h, fill: { color: colors.white }, line: { color: "E2E8F0", width: 0.8 }, shadow: { type: "outer", color: "000000", opacity: 0.10, blur: 3, angle: 45, distance: 1 } });
  slide.addShape("rect", { x, y, w: 0.08, h, fill: { color: accent }, line: { color: accent } });
  slide.addText(title, { x: x + 0.18, y: y + 0.14, w: w - 0.3, h: 0.22, fontFace: "Microsoft YaHei", fontSize: 11, bold: true, color: colors.navy, margin: 0 });
  slide.addText(body, { x: x + 0.18, y: y + 0.48, w: w - 0.3, h: h - 0.58, fontFace: "Microsoft YaHei", fontSize: 8.5, color: colors.muted, breakLine: false, margin: 0.02, fit: "shrink" });
}

function addBullets(slide, items, x, y, w, h, fontSize = 10) {
  slide.addText(items.map((text, idx) => ({ text, options: { bullet: true, breakLine: idx < items.length - 1 } })), {
    x, y, w, h, fontFace: "Microsoft YaHei", fontSize, color: colors.ink, margin: 0.03, breakLine: false, fit: "shrink",
  });
}

function buildPptx() {
  const pptx = new PptxGenJS();
  pptx.layout = "LAYOUT_WIDE";
  pptx.author = "Codex";
  pptx.subject = "SGDA EEG emotion recognition stage report";
  pptx.title = "SGDA 脑机接口情感识别阶段汇报";
  pptx.lang = "zh-CN";
  pptx.theme = {
    headFontFace: "Microsoft YaHei",
    bodyFontFace: "Microsoft YaHei",
    lang: "zh-CN",
  };

  let slide = pptx.addSlide();
  slide.background = { color: colors.navy };
  slide.addShape("rect", { x: 0, y: 0, w: 10, h: 5.625, fill: { color: colors.navy }, line: { color: colors.navy } });
  slide.addShape("rect", { x: 0.55, y: 0.58, w: 0.14, h: 3.95, fill: { color: colors.teal }, line: { color: colors.teal } });
  slide.addText("SGDA 脑机接口情感识别", { x: 0.9, y: 1.25, w: 8.2, h: 0.55, fontFace: "Microsoft YaHei", fontSize: 31, bold: true, color: colors.white, margin: 0 });
  slide.addText("阶段汇报：代码梳理、几何增强与实验结果", { x: 0.92, y: 2.02, w: 7.2, h: 0.32, fontFace: "Microsoft YaHei", fontSize: 15, color: "CCFBF1", margin: 0 });
  ["DEAP valence", "跨被试 LOSO", "SGDA + Riemann"].forEach((t, i) => addChip(slide, t, 0.92 + i * 1.7, 2.78, 1.35, i === 1 ? "E0F2FE" : colors.mint));
  slide.addText("2026-06-05", { x: 0.92, y: 4.82, w: 2.0, h: 0.2, fontFace: "Microsoft YaHei", fontSize: 10, color: "94A3B8", margin: 0 });

  slide = pptx.addSlide();
  addTitle(slide, "本阶段完成了什么", "从代码结构到实验结果，已经形成可复现实验链路");
  addCard(slide, 0.55, 1.15, 2.75, 1.25, "基线复现", "完成 DEAP valence 跨被试 SGDA 主流程，Acc 65.81%。");
  addCard(slide, 3.65, 1.15, 2.75, 1.25, "语义原型", "CLIP/BERT 等文本编码接口，把标签文本变成类别原型。", colors.amber);
  addCard(slide, 6.75, 1.15, 2.75, 1.25, "几何增强", "新增 Riemann 切空间与 Tangent Aux 辅助约束。", colors.green);
  addCard(slide, 0.55, 3.05, 2.75, 1.25, "多数据集", "覆盖 DEAP、SEED、SEED-IV、SEED-V、DREAMER。", colors.green);
  addCard(slide, 3.65, 3.05, 2.75, 1.25, "结果记录", "CSV 保存 acc、macro-F1、micro-F1 和均值方差。");
  addCard(slide, 6.75, 3.05, 2.75, 1.25, "问题定位", "几何辅助目前低于基线，已形成后续调参方向。", colors.red);
  addFooter(slide, 2);

  slide = pptx.addSlide();
  addTitle(slide, "代码结构", "项目已经从数据、模型、实验和评估四层展开");
  const stages = [
    ["数据读取", "load_data.py\npreprocess.py"],
    ["语义编码", "text_to_vector.py\nlabel mapper"],
    ["模型训练", "models/model.py\nexperiments/*.py"],
    ["评估保存", "mix_utils.py\nlog_utils.py"],
  ];
  stages.forEach((s, i) => {
    const x = 0.6 + i * 2.35;
    slide.addShape("rect", { x, y: 1.55, w: 1.75, h: 1.15, fill: { color: i % 2 ? "ECFEFF" : colors.mint }, line: { color: "99F6E4" } });
    slide.addText(s[0], { x: x + 0.15, y: 1.75, w: 1.45, h: 0.23, fontFace: "Microsoft YaHei", fontSize: 13, bold: true, color: colors.teal, align: "center", margin: 0 });
    slide.addText(s[1], { x: x + 0.15, y: 2.12, w: 1.45, h: 0.34, fontFace: "Consolas", fontSize: 7.4, color: colors.muted, align: "center", margin: 0 });
    if (i < stages.length - 1) {
      slide.addShape("line", { x: x + 1.85, y: 2.12, w: 0.75, h: 0, line: { color: colors.teal, width: 2, beginArrowType: "none", endArrowType: "triangle" } });
    }
  });
  addBullets(slide, [
    "配置层统一数据集、切分方式、标签边界和训练参数。",
    "训练脚本以目标被试为 held-out domain，其余被试作为多源域。",
    "模型输出进入语义原型空间，不依赖普通线性分类头。",
  ], 0.8, 3.55, 8.4, 0.85, 10);
  addFooter(slide, 3);

  slide = pptx.addSlide();
  addTitle(slide, "SGDA 基线机制", "共享提取、源域特定投影、语义对齐和自适应融合");
  addCard(slide, 0.65, 1.18, 2.1, 1.15, "SFE", "Conv2D 处理 [T,C,F] DE 特征，输出 512 维共享表征。");
  addCard(slide, 3.0, 1.18, 2.1, 1.15, "DSFE", "每个源被试一个投影分支，保留个体域差异。", colors.amber);
  addCard(slide, 5.35, 1.18, 2.1, 1.15, "语义原型", "positive/negative 经 CLIP 编码为归一化类别向量。", colors.green);
  addCard(slide, 7.7, 1.18, 1.85, 1.15, "融合", "按 source centroid 距离做 sample-wise 加权。", colors.teal);
  addBullets(slide, [
    "align_loss：源域投影向对应文本原型靠近。",
    "mmd_linear：缩小源域和目标域的分布差异。",
    "discrepancy_loss：约束多个源分支对目标样本输出一致。",
    "get_preds：评估阶段用源域 centroid 计算自适应权重。",
  ], 0.78, 3.15, 8.5, 1.05, 10.2);
  addFooter(slide, 4);

  slide = pptx.addSlide();
  addTitle(slide, "几何增强设计", "Tangent Aux 保留主路径，只在训练时提供几何约束");
  slide.addShape("rect", { x: 0.7, y: 1.25, w: 8.6, h: 2.6, fill: { color: colors.pale }, line: { color: "E2E8F0" } });
  const flow = [
    ["DE 样本", "T×C×F"],
    ["SPD", "OAS covariance"],
    ["Tangent", "Riemann log-map"],
    ["Projector", "Linear to 512"],
    ["L_geo_aux", "MSE with SFE"],
  ];
  flow.forEach((f, i) => {
    const x = 0.95 + i * 1.72;
    slide.addShape("roundRect", { x, y: 1.75, w: 1.25, h: 0.72, rectRadius: 0.06, fill: { color: i === 4 ? "FEE2E2" : colors.white }, line: { color: i === 4 ? "FCA5A5" : "99F6E4" } });
    slide.addText(f[0], { x: x + 0.08, y: 1.9, w: 1.09, h: 0.16, fontFace: "Microsoft YaHei", fontSize: 9.8, bold: true, color: i === 4 ? colors.red : colors.teal, align: "center", margin: 0 });
    slide.addText(f[1], { x: x + 0.08, y: 2.17, w: 1.09, h: 0.12, fontFace: "Consolas", fontSize: 6.4, color: colors.muted, align: "center", margin: 0 });
    if (i < flow.length - 1) {
      slide.addShape("line", { x: x + 1.29, y: 2.12, w: 0.38, h: 0, line: { color: colors.teal, width: 1.5, endArrowType: "triangle" } });
    }
  });
  addBullets(slide, [
    "训练损失：L_cls + alpha L_mmd + beta L_disc + 0.001 L_geo_aux。",
    "推理阶段仍复用基线 DE 路径，避免增加部署复杂度。",
    "关键风险是辅助几何目标与语义分类目标发生冲突。",
  ], 0.85, 4.25, 8.2, 0.65, 9.5);
  addFooter(slide, 5);

  slide = pptx.addSlide();
  addTitle(slide, "实验结果对比", "完整 DEAP 32 被试条件下，当前基线仍最好");
  const bars = [
    ["SGDA", 65.81, colors.teal],
    ["Riemann", 65.39, colors.green],
    ["DE-Riemann", 62.15, colors.amber],
    ["Tangent Aux", 60.68, colors.red],
    ["SPD Align", 59.68, "64748B"],
  ];
  const max = 70;
  bars.forEach((b, i) => {
    const y = 1.28 + i * 0.62;
    slide.addText(b[0], { x: 0.75, y: y + 0.05, w: 1.3, h: 0.16, fontFace: "Microsoft YaHei", fontSize: 9, color: colors.ink, margin: 0 });
    slide.addShape("rect", { x: 2.1, y, w: 5.6, h: 0.28, fill: { color: "E2E8F0" }, line: { color: "E2E8F0" } });
    slide.addShape("rect", { x: 2.1, y, w: 5.6 * (b[1] / max), h: 0.28, fill: { color: b[2] }, line: { color: b[2] } });
    slide.addText(`${b[1].toFixed(2)}%`, { x: 7.85, y: y + 0.04, w: 0.85, h: 0.14, fontFace: "Consolas", fontSize: 8.5, bold: true, color: colors.ink, margin: 0 });
  });
  addCard(slide, 0.75, 4.45, 8.3, 0.58, "阶段判断", "Tangent Aux 的 100% 只来自 3 个目标被试调试文件，不能作为最终结果；完整 32 被试 bs64 才是当前可信对比。", colors.red);
  addFooter(slide, 6);

  slide = pptx.addSlide();
  addTitle(slide, "为什么几何辅助暂时没有提升", "问题更像目标函数与实验协议的匹配问题");
  addCard(slide, 0.65, 1.2, 2.85, 1.25, "梯度方向冲突", "切空间 MSE 可能强调协方差结构还原，而非类别语义可分。", colors.red);
  addCard(slide, 3.78, 1.2, 2.85, 1.25, "坐标系不一致", "TangentSpace 被试内独立拟合，跨被试对齐参考点可能不同。", colors.amber);
  addCard(slide, 6.91, 1.2, 2.25, 1.25, "尺度敏感", "lambda 小不代表梯度小，需要看 loss 量级。", colors.teal);
  addBullets(slide, [
    "下一步不能只继续加模块，应先统一实验协议和 loss 监控。",
    "重点比较 lambda=0 与不同 lambda 的差异，确认几何约束是否真有信息增益。",
    "Riemann 输入模型接近基线，说明几何特征本身有价值，但融合方式还需要重构。",
  ], 0.85, 3.35, 8.2, 0.95, 10);
  addFooter(slide, 7);

  slide = pptx.addSlide();
  addTitle(slide, "下一阶段计划", "先把可比较性做扎实，再优化方法");
  const plans = [
    ["1", "统一协议", "32 被试、同 batch size、同 epoch、同 seed。"],
    ["2", "lambda 扫描", "0/1e-5/1e-4/1e-3/1e-2，记录各 loss 量级。"],
    ["3", "几何坐标", "尝试共享 TangentSpace 参考点或源域统一拟合。"],
    ["4", "消融实验", "无 MMD、无 discrepancy、average/adaptive fusion。"],
    ["5", "文档整理", "更新 README 与运行说明，沉淀复现实验流程。"],
  ];
  plans.forEach((p, i) => {
    const y = 1.15 + i * 0.72;
    slide.addShape("oval", { x: 0.78, y, w: 0.38, h: 0.38, fill: { color: colors.teal }, line: { color: colors.teal } });
    slide.addText(p[0], { x: 0.78, y: y + 0.08, w: 0.38, h: 0.12, fontFace: "Consolas", fontSize: 8.5, bold: true, color: colors.white, align: "center", margin: 0 });
    slide.addText(p[1], { x: 1.35, y: y + 0.03, w: 1.55, h: 0.18, fontFace: "Microsoft YaHei", fontSize: 11.5, bold: true, color: colors.navy, margin: 0 });
    slide.addText(p[2], { x: 3.0, y: y + 0.05, w: 6.0, h: 0.16, fontFace: "Microsoft YaHei", fontSize: 9.5, color: colors.muted, margin: 0 });
  });
  addFooter(slide, 8);

  slide = pptx.addSlide();
  slide.background = { color: colors.navy };
  slide.addText("阶段结论", { x: 0.65, y: 0.72, w: 3.0, h: 0.48, fontFace: "Microsoft YaHei", fontSize: 30, bold: true, color: colors.white, margin: 0 });
  slide.addText("当前代码已经具备完整跨被试 SGDA 实验链路；几何增强方向有探索价值，但完整 32 被试结果尚未超过基线。下一阶段的关键不是继续堆叠模块，而是统一协议、定位损失冲突并做系统消融。", {
    x: 0.72, y: 1.8, w: 8.4, h: 1.15, fontFace: "Microsoft YaHei", fontSize: 18, color: "E2E8F0", margin: 0.04, breakLine: false, fit: "shrink",
  });
  slide.addShape("rect", { x: 0.72, y: 3.7, w: 2.45, h: 0.08, fill: { color: colors.teal }, line: { color: colors.teal } });
  slide.addText("建议汇报重点：基线已跑通，几何辅助有负向结果，下一步用消融和统一协议把问题收束。", {
    x: 0.72, y: 4.05, w: 7.8, h: 0.3, fontFace: "Microsoft YaHei", fontSize: 11, color: "99F6E4", margin: 0,
  });
  addFooter(slide, 9);

  return pptx.writeFile({ fileName: pptPath });
}

(async () => {
  await buildDocx();
  await buildPptx();
  console.log(reportPath);
  console.log(pptPath);
})();
