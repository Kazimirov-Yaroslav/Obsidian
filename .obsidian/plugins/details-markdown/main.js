"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// src/main.ts
var main_exports = {};
__export(main_exports, {
  default: () => DetailsMarkdownPlugin
});
module.exports = __toCommonJS(main_exports);
var import_obsidian = require("obsidian");

// src/DetailsTagPatterns.ts
var DetailsTagPatterns = class {
};
// WHAT (non-obvious regexes): opening tag on its own line, allowing any attributes
// (e.g. `open`, `class="..."`); single-line summary (attributes allowed) capturing its
// inner content; bare closing tag on its own line. `\s[^>]*` requires a space before the
// attribute list and stops at the first `>`, so `<detailsfoo>` is not matched.
DetailsTagPatterns.OPENING_LINE = /^<details(?:\s[^>]*)?>\s*$/i;
DetailsTagPatterns.SUMMARY_LINE = /^<summary(?:\s[^>]*)?>(.*)<\/summary>\s*$/i;
DetailsTagPatterns.CLOSING_LINE = /^<\/details>\s*$/i;
/** A fenced-code delimiter line (``` or ~~~, up to 3 leading spaces, per CommonMark). */
DetailsTagPatterns.FENCE_DELIMITER_LINE = /^ {0,3}(?:`{3,}|~{3,})/;

// src/DetailsBlockParser.ts
var DetailsBlockParser = class _DetailsBlockParser {
  /**
   * Parses a source snippet expected to contain exactly one supported block
   * (blank lines around the block are tolerated). Returns null for any
   * unsupported or malformed shape.
   */
  static parse(source) {
    const lines = _DetailsBlockParser.trimSurroundingBlankLines(source.split("\n"));
    if (lines.length < 2) {
      return null;
    }
    if (!DetailsTagPatterns.OPENING_LINE.test(lines[0])) {
      return null;
    }
    if (!DetailsTagPatterns.CLOSING_LINE.test(lines[lines.length - 1])) {
      return null;
    }
    const innerLines = lines.slice(1, lines.length - 1);
    const summaryMatch = innerLines.length > 0 ? innerLines[0].match(DetailsTagPatterns.SUMMARY_LINE) : null;
    if (summaryMatch === null && innerLines.length > 0 && /^<summary/i.test(innerLines[0].trim())) {
      return null;
    }
    const bodyLines = summaryMatch ? innerLines.slice(1) : innerLines;
    return {
      summaryText: summaryMatch ? summaryMatch[1] : null,
      bodyMarkdown: bodyLines.join("\n")
    };
  }
  static trimSurroundingBlankLines(lines) {
    let start = 0;
    let end = lines.length;
    while (start < end && lines[start].trim() === "") start++;
    while (end > start && lines[end - 1].trim() === "") end--;
    return lines.slice(start, end);
  }
};

// src/DetailsRangeScanner.ts
var DetailsRangeScanner = class _DetailsRangeScanner {
  static scan(text) {
    const lines = text.split("\n");
    const inFence = _DetailsRangeScanner.computeFenceMask(lines);
    const ranges = [];
    let i = 0;
    while (i < lines.length) {
      if (!inFence[i] && DetailsTagPatterns.OPENING_LINE.test(lines[i])) {
        const endLine = _DetailsRangeScanner.findMatchingClose(lines, inFence, i);
        if (endLine !== -1) {
          ranges.push({ startLine: i, endLine });
          i = endLine + 1;
          continue;
        }
      }
      i++;
    }
    return ranges;
  }
  /** Line index of the close matching the opening at `openLine`, or -1 if unclosed. */
  static findMatchingClose(lines, inFence, openLine) {
    let depth = 1;
    for (let j = openLine + 1; j < lines.length; j++) {
      if (inFence[j]) {
        continue;
      }
      if (DetailsTagPatterns.OPENING_LINE.test(lines[j])) {
        depth++;
      } else if (DetailsTagPatterns.CLOSING_LINE.test(lines[j])) {
        depth--;
        if (depth === 0) {
          return j;
        }
      }
    }
    return -1;
  }
  /** True for lines inside (or delimiting) a fenced code block, so tags there are ignored. */
  static computeFenceMask(lines) {
    const mask = new Array(lines.length);
    let inFence = false;
    for (let i = 0; i < lines.length; i++) {
      if (DetailsTagPatterns.FENCE_DELIMITER_LINE.test(lines[i])) {
        mask[i] = true;
        inFence = !inFence;
      } else {
        mask[i] = inFence;
      }
    }
    return mask;
  }
};

// src/SectionRoleClassifier.ts
var SectionRoleClassifier = class {
  static classify(ranges, lineStart, lineEnd) {
    for (const range of ranges) {
      if (lineStart === range.startLine) {
        return { kind: "opening", range };
      }
      const startsInside = lineStart > range.startLine && lineStart <= range.endLine;
      if (startsInside && lineEnd <= range.endLine) {
        return { kind: "fragment", range };
      }
    }
    return { kind: "none" };
  }
};

// src/main.ts
var DEFAULT_SETTINGS = {
  enabled: true
};
var RENDERED_ATTRIBUTE = "data-details-markdown-rendered";
var HIDDEN_FRAGMENT_CLASS = "details-markdown-hidden-fragment";
var FRAGMENT_SWEEP_MAX_TRIES = 10;
var DetailsMarkdownPlugin = class extends import_obsidian.Plugin {
  constructor() {
    super(...arguments);
    this.settings = DEFAULT_SETTINGS;
    this.blocksByPath = /* @__PURE__ */ new Map();
  }
  async onload() {
    await this.loadSettings();
    this.addSettingTab(new DetailsMarkdownSettingTab(this.app, this));
    this.registerMarkdownPostProcessor(async (el, ctx) => {
      if (!this.settings.enabled) {
        return;
      }
      try {
        await this.processSection(el, ctx);
      } catch (error) {
        console.error("details-markdown: failed to process section", error);
      }
    });
  }
  onunload() {
    this.blocksByPath.clear();
    this.rerenderOpenMarkdownViews();
  }
  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }
  async saveSettings() {
    await this.saveData(this.settings);
  }
  /** Applies a settings toggle immediately: full re-render restores native or rendered state. */
  onEnabledSettingChanged() {
    this.blocksByPath.clear();
    this.rerenderOpenMarkdownViews();
  }
  rerenderOpenMarkdownViews() {
    for (const leaf of this.app.workspace.getLeavesOfType("markdown")) {
      const view = leaf.view;
      if (view instanceof import_obsidian.MarkdownView) {
        view.previewMode?.rerender(true);
      }
    }
  }
  rerenderViewsForPath(path) {
    for (const leaf of this.app.workspace.getLeavesOfType("markdown")) {
      const view = leaf.view;
      if (view instanceof import_obsidian.MarkdownView && view.file?.path === path) {
        view.previewMode?.rerender(true);
      }
    }
  }
  // ---------------------------------------------------------------------------
  // Section processing
  // ---------------------------------------------------------------------------
  async processSection(el, ctx) {
    const sectionInfo = ctx.getSectionInfo(el);
    if (sectionInfo === null) {
      return;
    }
    const lines = sectionInfo.text.split("\n");
    const ranges = DetailsRangeScanner.scan(sectionInfo.text);
    await this.reconcileTrackedBlocks(ctx, ranges, lines);
    const role = SectionRoleClassifier.classify(ranges, sectionInfo.lineStart, sectionInfo.lineEnd);
    if (role.kind === "opening") {
      await this.renderOpeningSection(el, ctx, role.range, lines);
    } else if (role.kind === "fragment") {
      this.hideFragmentIfBlockRendered(el, ctx, role.range);
    }
  }
  /** Renders the full block body (from raw source) into the section's <details> element. */
  async renderOpeningSection(sectionEl, ctx, range, lines) {
    const detailsEl = this.findSingleTopLevelDetails(sectionEl);
    if (detailsEl === null || detailsEl.hasAttribute(RENDERED_ATTRIBUTE)) {
      return;
    }
    const source = this.sliceRange(lines, range);
    const parsed = DetailsBlockParser.parse(source);
    if (parsed === null || parsed.bodyMarkdown === "") {
      return;
    }
    const summaryEl = detailsEl.querySelector(":scope > summary");
    detailsEl.empty();
    if (summaryEl !== null) {
      detailsEl.appendChild(summaryEl);
    }
    const bodyContainer = detailsEl.createDiv({ cls: "details-markdown-body" });
    const lifecycleOwner = new import_obsidian.MarkdownRenderChild(bodyContainer);
    ctx.addChild(lifecycleOwner);
    detailsEl.setAttribute(RENDERED_ATTRIBUTE, "true");
    await import_obsidian.MarkdownRenderer.render(
      this.app,
      parsed.bodyMarkdown,
      bodyContainer,
      ctx.sourcePath,
      lifecycleOwner
    );
    this.pathBlocks(ctx.sourcePath).rendered.push({
      sectionEl,
      detailsEl,
      bodyContainer,
      lifecycleOwner,
      renderedSource: source,
      everConnected: sectionEl.isConnected
    });
    this.scheduleFragmentSweep(sectionEl, ctx, FRAGMENT_SWEEP_MAX_TRIES);
  }
  /**
   * Hides an escaped fragment section, but only when its block's opening is actually
   * rendered — otherwise (e.g. opening glued into a list section) hiding would lose content.
   */
  hideFragmentIfBlockRendered(el, ctx, range) {
    const opening = this.findRenderedOpening(ctx, ctx.sourcePath, range.startLine);
    if (opening === null) {
      return;
    }
    this.hideFragment(el, ctx.sourcePath);
  }
  /** After the opening attaches, hides already-rendered sibling sections inside its block. */
  scheduleFragmentSweep(sectionEl, ctx, triesLeft) {
    if (triesLeft <= 0) {
      return;
    }
    requestAnimationFrame(() => {
      try {
        if (!sectionEl.isConnected) {
          this.scheduleFragmentSweep(sectionEl, ctx, triesLeft - 1);
          return;
        }
        this.sweepSiblingFragments(sectionEl, ctx);
      } catch (error) {
        console.error("details-markdown: fragment sweep failed", error);
      }
    });
  }
  sweepSiblingFragments(sectionEl, ctx) {
    const openingInfo = ctx.getSectionInfo(sectionEl);
    const parent = sectionEl.parentElement;
    if (openingInfo === null || parent === null) {
      return;
    }
    const ranges = DetailsRangeScanner.scan(openingInfo.text);
    for (const sibling of Array.from(parent.children)) {
      if (sibling === sectionEl || !(sibling instanceof HTMLElement) || sibling.classList.contains(HIDDEN_FRAGMENT_CLASS)) {
        continue;
      }
      const info = ctx.getSectionInfo(sibling);
      if (info === null) {
        continue;
      }
      const role = SectionRoleClassifier.classify(ranges, info.lineStart, info.lineEnd);
      if (role.kind === "fragment" && role.range.startLine === openingInfo.lineStart) {
        this.hideFragment(sibling, ctx.sourcePath);
      }
    }
  }
  // ---------------------------------------------------------------------------
  // Reconciliation: staleness after interior edits, unhiding, dead-entry cleanup
  // ---------------------------------------------------------------------------
  async reconcileTrackedBlocks(ctx, ranges, lines) {
    const path = ctx.sourcePath;
    const blocks = this.blocksByPath.get(path);
    if (blocks === void 0) {
      return;
    }
    for (const entry of [...blocks.rendered]) {
      entry.everConnected || (entry.everConnected = entry.sectionEl.isConnected);
      if (entry.everConnected && !entry.sectionEl.isConnected) {
        this.remove(blocks.rendered, entry);
        continue;
      }
      const info = ctx.getSectionInfo(entry.sectionEl);
      if (info === null) {
        continue;
      }
      const range = ranges.find((r) => r.startLine === info.lineStart);
      if (range === void 0) {
        this.remove(blocks.rendered, entry);
        this.rerenderViewsForPath(path);
        continue;
      }
      const source = this.sliceRange(lines, range);
      if (source !== entry.renderedSource) {
        await this.rerenderStaleBody(entry, source, ctx);
      }
    }
    for (const fragment of [...blocks.hidden]) {
      fragment.everConnected || (fragment.everConnected = fragment.el.isConnected);
      if (fragment.everConnected && !fragment.el.isConnected) {
        this.remove(blocks.hidden, fragment);
        continue;
      }
      const info = ctx.getSectionInfo(fragment.el);
      if (info === null) {
        continue;
      }
      const role = SectionRoleClassifier.classify(ranges, info.lineStart, info.lineEnd);
      if (role.kind !== "fragment") {
        fragment.el.classList.remove(HIDDEN_FRAGMENT_CLASS);
        this.remove(blocks.hidden, fragment);
      }
    }
  }
  /**
   * Re-renders a block body in place after an interior (blank-line-separated) edit:
   * Obsidian re-renders only the edited fragment section, never the opening one.
   */
  async rerenderStaleBody(entry, source, ctx) {
    const parsed = DetailsBlockParser.parse(source);
    if (parsed === null || parsed.bodyMarkdown === "") {
      this.remove(this.pathBlocks(ctx.sourcePath).rendered, entry);
      this.rerenderViewsForPath(ctx.sourcePath);
      return;
    }
    entry.lifecycleOwner.unload();
    entry.bodyContainer.empty();
    const lifecycleOwner = new import_obsidian.MarkdownRenderChild(entry.bodyContainer);
    ctx.addChild(lifecycleOwner);
    entry.lifecycleOwner = lifecycleOwner;
    entry.renderedSource = source;
    await import_obsidian.MarkdownRenderer.render(
      this.app,
      parsed.bodyMarkdown,
      entry.bodyContainer,
      ctx.sourcePath,
      lifecycleOwner
    );
  }
  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------
  findRenderedOpening(ctx, path, blockStartLine) {
    for (const entry of this.pathBlocks(path).rendered) {
      const info = ctx.getSectionInfo(entry.sectionEl);
      if (info !== null && info.lineStart === blockStartLine) {
        return entry;
      }
    }
    return null;
  }
  hideFragment(el, path) {
    el.classList.add(HIDDEN_FRAGMENT_CLASS);
    this.pathBlocks(path).hidden.push({ el, everConnected: el.isConnected });
  }
  /**
   * Returns the section's <details> element only when the section contains exactly
   * one top-level <details>; otherwise the section is not a supported opening.
   */
  findSingleTopLevelDetails(el) {
    const topLevel = Array.from(el.querySelectorAll("details")).filter(
      (d) => d.parentElement?.closest("details") === null
    );
    return topLevel.length === 1 ? topLevel[0] : null;
  }
  sliceRange(lines, range) {
    return lines.slice(range.startLine, range.endLine + 1).join("\n");
  }
  pathBlocks(path) {
    let blocks = this.blocksByPath.get(path);
    if (blocks === void 0) {
      blocks = { rendered: [], hidden: [] };
      this.blocksByPath.set(path, blocks);
    }
    return blocks;
  }
  remove(list, item) {
    const index = list.indexOf(item);
    if (index !== -1) {
      list.splice(index, 1);
    }
  }
};
var DetailsMarkdownSettingTab = class extends import_obsidian.PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }
  display() {
    this.containerEl.empty();
    new import_obsidian.Setting(this.containerEl).setName("Render Markdown inside <details> blocks").setDesc("When off, Obsidian's native (literal) behavior is restored.").addToggle(
      (toggle) => toggle.setValue(this.plugin.settings.enabled).onChange(async (value) => {
        this.plugin.settings.enabled = value;
        await this.plugin.saveSettings();
        this.plugin.onEnabledSettingChanged();
      })
    );
  }
};

/* nosourcemap */