"use client";

import { ChangeEvent, ClipboardEvent, DragEvent, PointerEvent as ReactPointerEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = "http://127.0.0.1:8765";

type View = "chat" | "learning" | "spaces" | "library" | "images" | "skills" | "tools" | "memory" | "lab" | "infra" | "settings";
type ProviderStatus = {
  provider: string;
  label: string;
  capability: string;
  configured: boolean;
  model: string;
  connection_status: "connected" | "failed" | "not_tested" | "not_configured";
  last_checked_at?: string | null;
  error_code?: string | null;
};
type Space = { id: string; name: string; color: string; document_count: number; chunk_count: number };
type DocumentItem = {
  id: string;
  original_name: string;
  file_type: string;
  title: string;
  summary: string;
  index_status: string;
  updated_at: string;
  space_id: string;
  effective_index_status?: string;
  library_copy_exists?: boolean;
  chunk_count?: number;
  embedding_count?: number;
};
type StagedFile = {
  id: string;
  original_name: string;
  file_type: string;
  size_bytes: number;
  title: string;
  summary: string;
  tags: string[];
  metadata_source: string;
  sections: number;
};
type Citation = {
  id: number; title: string; file: string; locator: string; chunk_id: string;
  kind?: "document" | "web"; url?: string | null; site_name?: string | null;
};
type MemorySuggestion = { id: string; kind: string; content: string; status: string };
type PlanTaskStatus = "pending" | "in_progress" | "completed" | "awaiting_confirmation" | "failed";
type AgentPlan = {
  version: number; planner: string; goal: string; intent: string; skill?: string | null; route_reason?: string; status?: string;
  fallback_skill?: string; fallback_reason?: string; retrieval_origin?: string;
  tasks: Array<{ id: string; title: string; status: PlanTaskStatus; source: string; detail?: string }>;
  grounding?: { status: string; risk: "low" | "medium" | "high" | "unknown"; label: string; explanation: string };
  tool_calls?: Array<{ name?: string; status?: string; duration_ms?: number; result_count?: number; error_code?: string; recoverable?: boolean }>;
  explanation?: string;
};
type ChatMessage = { id: string; role: "user" | "assistant"; content: string; citations?: Citation[]; memorySuggestion?: MemorySuggestion | null; plan?: AgentPlan | null };
type SourcePreview = Citation & { text?: string; heading?: string };
type MemoryItem = { id: string; kind: string; content: string; source: string; status: string; updated_at: string; version?: number; conflict_with_id?: string | null };
type MemoryDetail = MemoryItem & { events: Array<{ id: string; event_type: string; version: number; source: string; created_at: string }>; recent_recalls: Array<{ id: string; reason: string; score: number; query: string; created_at: string }> };
type ShortTermMemory = {
  conversation: ConversationItem | null;
  messages: Array<{ id: string; role: "user" | "assistant"; content: string; created_at: string }>;
  working_facts: Array<{ kind: string; content: string; status: "short_term" | "pending" | "enabled" | "disabled" }>;
  message_count: number;
  scope: string;
  window_size: number;
  previous_message_limit: number;
  storage_policy: string;
  expires?: string;
};
type ConversationItem = { id: string; title: string; space_id: string; preview?: string; message_count: number; updated_at: string };
type ImageAsset = {
  document_id: string; title: string; original_name: string; description: string; ocr_text: string;
  tags: string[]; width: number; height: number; score: number; updated_at: string;
};
type DocumentStatus = {
  id: string; title: string; original_name: string; status: string; library_copy_exists: boolean;
  library_path: string; chunk_count: number; embedding_count: number; embedding_model?: string | null;
  updated_at: string; latest_job?: IndexJob | null;
};
type Health = { status: string; chat_ready: boolean; embedding_ready: boolean; rerank_ready?: boolean; chat_model: string; embedding_model: string; data_dir?: string };
type IndexJob = {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  phase: string;
  progress: number;
  total: number;
  completed: number;
  message: string;
  error_message?: string | null;
};
type SkillItem = {
  id: string; name: string; description: string; tools: string[]; read_scope: string[]; write_scope: string[];
  timeout_seconds?: number | null; requires_confirmation_for_write: boolean; source: "builtin" | "user";
  has_skill_md: boolean; content: string; steps: string[]; recoverable_errors: string[]; output_fields: string[];
};
type EvaluationCase = {
  id: string; question: string; expected_document_id: string; expected_document_title: string;
  expected_original_name: string; expected_locator: string; updated_at: string;
};
type EvaluationRun = {
  id: string; top_k: number; case_count: number; recall: number; mrr: number; ndcg: number;
  mean_latency_ms: number; p95_latency_ms: number;
  details: Array<{ case_id: string; question: string; hit: boolean; rank: number | null; latency_ms: number; returned: Array<{ title: string; locator: string; score: number }> }>;
};
type AgentQualityResult = {
  target: number; publish_ready: boolean; publication_blockers: string[]; notes: string;
  metrics: Record<string, { value: number | null; case_count: number; target: number; status?: string }>;
  route_details: Array<{ question: string; expected_skill: string; actual_skill: string; passed: boolean }>;
  refusal_details: Array<{ question: string; passed: boolean; behavior: string }>;
};
type ToolDefinition = {
  name: string; description: string; read_scopes: string[]; write_scopes: string[]; network_scopes: string[];
  timeout_seconds: number; confirmation_required: boolean; availability: string; unavailable_reason?: string | null;
  parameter_summary?: Array<{ name: string; type: string; required: boolean }>;
};
type ToolRun = {
  id: string; conversation_id?: string | null; tool_name: string; status: string;
  error_code?: string | null; duration_ms?: number | null; created_at: string;
  input_summary?: Record<string, unknown>; output_summary?: Record<string, unknown>;
};
type RoutePreview = {
  question: string; intent: string; skill: string; tools: string[]; plan?: AgentPlan;
  memory_candidate?: { kind: string; content: string; status: string } | null;
};
type BackupItem = { id: string; created_at: string; reason: string; size_bytes: number; status: string };
type StorageInfo = {
  data_dir: string;
  database_bytes: number;
  folders: Record<string, number>;
  total_bytes: number;
  counts: { documents: number; chunks: number; conversations: number; memories: number };
  backups: BackupItem[];
};
type PrivacySettings = {
  web_search_enabled: boolean;
  cloud_document_analysis_enabled: boolean;
  cloud_image_analysis_enabled: boolean;
  memory_suggestions_enabled: boolean;
  sensitive_data_protection_enabled: boolean;
  fixed_boundaries: Record<string, string>;
};
type CapabilityStatus = "verified" | "partial" | "planned";
type LearningOverview = {
  generated_at: string;
  documents: { total: number; ready: number; failed: number; chunks: number };
  memories: { pending: number; enabled: number; disabled: number; events: number; recalls: number; conflicts: number };
  tools: { total: number; succeeded: number; failed: number; average_duration_ms?: number | null; available: string[]; all: string[] };
  skills: string[];
  evaluation: { case_count: number; latest?: EvaluationRun | null };
  context_policy: { model_hard_limit: number; input_budget: number; safety_margin: number; output_reserve: number; effective_input_limit: number; estimator: string };
  agent: { trace_count: number; completed: number };
  workflow: Array<{ id: string; label: string; plain: string; status: CapabilityStatus; route: View }>;
  capability_matrix: Array<{ id: string; topic: string; capability: string; status: CapabilityStatus; evidence: string; verify: string; route: View; next: string }>;
};
type AgentTrace = {
  id: string; conversation_id?: string | null; intent?: string | null; selected_skill?: string | null; status: string;
  context_tokens: number; context_budget: number; summary_version?: number | null; retrieval_count: number; citation_count: number;
  exposed_tool_count: number; schema_token_estimate: number; error_type?: string | null; started_at: string; duration_ms?: number | null;
  stages: Array<{ id: string; stage_name: string; status: string; duration_ms: number; result_summary: Record<string, unknown>; error_type?: string | null }>;
};
type InfraJob = {
  id: string; job_type: string; status: string; phase: string; progress: number; message: string;
  attempt: number; max_attempts: number; error_code?: string | null; created_at: string; updated_at: string;
};
type InfraTrace = {
  id: string; trace_type: string; name: string; status: string; duration_ms?: number | null; started_at: string;
  span_count?: number; failed_span_count?: number; attributes?: Record<string, unknown>;
  spans?: Array<{ id: string; operation: string; kind: string; status: string; duration_ms?: number | null; started_at: string; attributes: Record<string, unknown> }>;
};
type IndexGeneration = {
  id: string; space_id: string; status: string; is_active: number; provider: string; model: string; dimension: number;
  strategy: string; chunk_size: number; chunk_overlap: number; vector_count: number; index_bytes: number; created_at: string;
  estimate?: { chunk_count: number; cache_hits: number; cache_misses: number; cache_hit_rate: number; estimated_batches: number; estimated_input_characters: number; cost_status: string; requires_confirmation: boolean };
};
type EvalDataset = {
  id: string; name: string; version: string; status: string; case_count: number; accepted_count?: number; draft_count?: number; rejected_count?: number;
  cases?: Array<{ id: string; question: string; status: string; split: string; query_type: string; difficulty: string; gold: Array<{ title: string; locator: string; relevance: number }> }>;
};
type Experiment = {
  id: string; name: string; status: string; dataset_version_id: string; created_at: string; finished_at?: string | null; config: Record<string, unknown>;
  config_hash?: string; git_revision?: string; machine?: Record<string, unknown>;
  summary: { case_count?: number; document_recall?: Record<string, number>; evidence_recall?: Record<string, number>; mrr?: number; ndcg_10?: number; citation_resolvable_rate?: number; latency_ms?: Record<string, number>; failure_counts?: Record<string, number>; query_types?: Record<string, { evidence_recall: number; case_count: number }>; dataset?: { name?: string; version?: string; content_hash?: string } };
  cases?: Array<{ case_id: string; question: string; failure_category?: string | null; latency_ms: number; metrics: Record<string, unknown>; rankings: { returned?: Array<Record<string, unknown>> } }>;
};
type InfraOverview = {
  generated_at: string; jobs: Record<string, number>; traces: { total: number; succeeded: number; failed: number; average_ms?: number | null; p95_ms?: number | null };
  indexes: IndexGeneration[]; recent_traces: InfraTrace[]; providers: ProviderStatus[];
};
type DuelResult = {
  question: string;
  left: { trace_id: string; duration_ms: number; stages: Array<{ stage: string; duration_ms: number; count: number }>; results: Array<Record<string, any>> };
  right: { trace_id: string; duration_ms: number; stages: Array<{ stage: string; duration_ms: number; count: number }>; results: Array<Record<string, any>> };
  rank_movement: Array<{ chunk_id: string; left_rank?: number; right_rank?: number; delta: number }>;
};
type RegressionResult = {
  status: string; checks: Array<{ name: string; baseline: number; candidate: number; delta: number; status: string; rule: string }>;
  confidence: { method: string; samples: number; evidence_recall_delta_95_ci: [number, number] };
};
type PerformanceBenchmark = {
  id: string; status: string; created_at: string; config: Record<string, any>;
  result: { quality_claim?: boolean; note?: string; results?: Array<Record<string, any>> };
};
type DocumentCloudPolicy = {
  document_id: string; title: string; original_name: string; file_type: string;
  embedding_allowed: number; llm_allowed: number; updated_at?: string | null;
};
type InfraBudget = { max_api_requests_per_run: number; max_embedding_input_characters: number; allow_multi_model_rebuild: boolean };

const NAV: Array<{ id: View; icon: string; label: string }> = [
  { id: "learning", icon: "◎", label: "Agent 透视" },
  { id: "spaces", icon: "◫", label: "知识空间" },
  { id: "library", icon: "▤", label: "文件资料库" },
  { id: "images", icon: "▧", label: "图片搜索" },
  { id: "skills", icon: "✦", label: "Skills" },
  { id: "tools", icon: "⌘", label: "Tools" },
  { id: "memory", icon: "◇", label: "Memory" },
  { id: "lab", icon: "⌁", label: "RAG 实验室" },
  { id: "infra", icon: "⌬", label: "AI Infra" },
];

function formatSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(value: string) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("zh-CN", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function experimentQualityScore(item: Experiment) {
  const summary = item.summary || {};
  return (summary.evidence_recall?.["5"] || 0) * 0.5 + (summary.mrr || 0) * 0.3 + (summary.ndcg_10 || 0) * 0.2;
}

function experimentPipelineLabel(item: Experiment) {
  const pipeline = String(item.config?.pipeline || "bm25");
  return ({ bm25: "BM25", dense: "Dense", hybrid: "Hybrid + RRF", hybrid_rerank: "Hybrid + Rerank" } as Record<string, string>)[pipeline] || pipeline;
}

function metricPercent(value?: number) {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "—";
}

function metricNumber(value?: number, digits = 3) {
  return typeof value === "number" ? value.toFixed(digits) : "—";
}

function metricMillis(value?: number) {
  return typeof value === "number" ? `${Math.round(value)} ms` : "—";
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new Error("无法连接本地 Agent 服务。请重新双击“启动-KUN.cmd”，等待浏览器自动打开后再试。");
  }
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const body = await response.json();
      message = typeof body.detail === "string" ? body.detail : body.detail?.message || message;
    } catch {
      // Keep the safe HTTP message.
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export default function Home() {
  const [view, setView] = useState<View>("chat");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [onboarding, setOnboarding] = useState(() => localStorage.getItem("kun.onboarding.complete") !== "1");
  const [onboardingStep, setOnboardingStep] = useState(0);
  const [health, setHealth] = useState<Health | null>(null);
  const [providers, setProviders] = useState<ProviderStatus[]>([]);
  const [spaces, setSpaces] = useState<Space[]>([]);
  const [selectedSpaceId, setSelectedSpaceId] = useState(() => localStorage.getItem("kun.activeSpace") || "ai-agent-learning");
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [memories, setMemories] = useState<MemoryItem[]>([]);
  const [conversations, setConversations] = useState<ConversationItem[]>([]);
  const [conversationQuery, setConversationQuery] = useState("");
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [message, setMessage] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [uploadError, setUploadError] = useState("");
  const [staged, setStaged] = useState<StagedFile[]>([]);
  const [imported, setImported] = useState(false);
  const [source, setSource] = useState<SourcePreview | null>(null);
  const [selectedDocumentStatus, setSelectedDocumentStatus] = useState<DocumentStatus | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const activeSpace = spaces.find((space) => space.id === selectedSpaceId) || spaces[0];
  const viewTitle = useMemo(() => ({
    chat: activeSpace?.name || "个人知识助理",
    learning: "Agent 透视",
    spaces: "知识空间",
    library: "文件资料库",
    images: "图片搜索",
    skills: "Skill 中心",
    tools: "Tool 中心",
    memory: "Memory 中心",
    lab: "RAG 实验室",
    infra: "AI Infra 控制台",
    settings: "设置",
  })[view], [view, activeSpace]);

  async function loadRuntime() {
    try {
      const [healthData, providerData, spaceData, documentData, memoryData] = await Promise.all([
        api<Health>("/api/health"),
        api<ProviderStatus[]>("/api/settings/providers"),
        api<Space[]>("/api/spaces"),
        api<DocumentItem[]>("/api/documents"),
        api<MemoryItem[]>("/api/memories"),
      ]);
      setHealth(healthData);
      setProviders(providerData);
      setSpaces(spaceData);
      setDocuments(documentData);
      setMemories(memoryData);
    } catch {
      setHealth(null);
    }
  }

  useEffect(() => {
    void loadRuntime();
    void searchConversations("");
    const refresh = window.setInterval(() => void loadRuntime(), 5000);
    return () => window.clearInterval(refresh);
  }, []);

  useEffect(() => {
    if (activeSpace && activeSpace.id !== selectedSpaceId) setSelectedSpaceId(activeSpace.id);
  }, [activeSpace, selectedSpaceId]);

  function chooseSpace(spaceId: string) {
    setSelectedSpaceId(spaceId);
    localStorage.setItem("kun.activeSpace", spaceId);
  }

  function newChat() {
    setCurrentConversationId(null);
    setMessages([]);
    setStatusMessage("");
    setSource(null);
    setView("chat");
  }

  async function searchConversations(value: string) {
    setConversationQuery(value);
    try {
      setConversations(await api<ConversationItem[]>(`/api/conversations?q=${encodeURIComponent(value)}`));
    } catch {
      // Runtime polling will recover the list.
    }
  }

  async function openConversation(conversationId: string) {
    const detail = await api<{ conversation: ConversationItem; messages: Array<ChatMessage & { citations_json?: string }> }>(`/api/conversations/${conversationId}`);
    setCurrentConversationId(conversationId);
    chooseSpace(detail.conversation.space_id);
    setMessages(detail.messages.map((item) => ({ id: item.id, role: item.role, content: item.content, citations: item.citations || [], plan: item.plan?.tasks?.length ? item.plan : null })));
    setSource(null);
    setView("chat");
  }

  async function deleteConversation(conversationId: string) {
    if (!window.confirm("删除这条对话记录？资料和 Memory 不会被删除。")) return;
    await api(`/api/conversations/${conversationId}`, { method: "DELETE" });
    if (currentConversationId === conversationId) newChat();
    await searchConversations(conversationQuery);
  }

  async function testProvider(provider: string) {
    setProviders((items) => items.map((item) => item.provider === provider ? { ...item, connection_status: "not_tested" } : item));
    try {
      await api(`/api/settings/providers/${provider}/test`, { method: "POST" });
    } finally {
      await loadRuntime();
    }
  }

  async function stageFiles(files: File[]) {
    if (!files.length) return;
    setUploadOpen(true);
    setImported(false);
    setStaged([]);
    setUploadError("");
    setUploadProgress(null);
    setUploadStatus(`正在上传 ${files.length} 个文件`);
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
    try {
      setUploadStatus(files.some((file) => file.type.startsWith("image/"))
        ? "正在识别图片文字和画面，并生成标题与摘要"
        : "正在解析文档并生成标题、摘要和标签");
      const result = await api<StagedFile[]>("/api/documents/stage", { method: "POST", body });
      setStaged(result);
      setUploadStatus("");
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "文件处理失败");
      setUploadStatus("");
    }
  }

  async function chooseFiles(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    event.target.value = "";
    await stageFiles(files);
  }

  function normalizePastedFiles(files: File[]) {
    return files.map((file, index) => {
      if (file.name.includes(".")) return file;
      const extension = ({ "image/png": "png", "image/jpeg": "jpg" } as Record<string, string>)[file.type] || "bin";
      return new File([file], `粘贴内容-${Date.now()}-${index + 1}.${extension}`, { type: file.type, lastModified: file.lastModified });
    });
  }

  function pasteFiles(files: File[]) {
    void stageFiles(normalizePastedFiles(files));
  }

  function updateStaged(id: string, patch: Partial<StagedFile>) {
    setStaged((items) => items.map((item) => item.id === id ? { ...item, ...patch } : item));
  }

  async function confirmImport() {
    if (!staged.length) return;
    setUploadError("");
    setUploadProgress(0);
    setUploadStatus("正在创建后台索引任务");
    try {
      const jobIds: string[] = [];
      for (const item of staged) {
        const accepted = await api<{ job_id: string }>(`/api/documents/${item.id}/confirm`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            title: item.title,
            summary: item.summary,
            tags: item.tags,
            space_id: activeSpace?.id || "ai-agent-learning",
          }),
        });
        jobIds.push(accepted.job_id);
      }
      let finished = false;
      while (!finished) {
        const jobs = await Promise.all(jobIds.map((jobId) => api<IndexJob>(`/api/index-jobs/${jobId}`)));
        const failed = jobs.find((job) => job.status === "failed");
        if (failed) throw new Error(failed.error_message || "建立索引失败");
        const average = Math.round(jobs.reduce((sum, job) => sum + job.progress, 0) / jobs.length);
        const active = jobs.find((job) => job.status !== "completed") || jobs[jobs.length - 1];
        setUploadProgress(average);
        setUploadStatus(active.message);
        finished = jobs.every((job) => job.status === "completed");
        if (!finished) await new Promise((resolve) => window.setTimeout(resolve, 800));
      }
      setImported(true);
      setUploadStatus("");
      setUploadProgress(100);
      await loadRuntime();
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "建立索引失败");
      setUploadStatus("");
      setUploadProgress(null);
    }
  }

  async function submitMessage() {
    const text = message.trim();
    if (!text || statusMessage) return;
    const userMessage: ChatMessage = { id: crypto.randomUUID(), role: "user", content: text };
    setMessages((items) => [...items, userMessage]);
    setMessage("");
    setStatusMessage("正在理解你的问题");
    setSource(null);
    try {
      const response = await fetch(`${API_BASE}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: text,
          space_id: activeSpace?.id || "ai-agent-learning",
          conversation_id: currentConversationId,
        }),
      });
      if (!response.ok || !response.body) throw new Error("本地 Agent 服务暂时不可用");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop() || "";
        for (const block of blocks) {
          const line = block.split("\n").find((item) => item.startsWith("data: "));
          if (!line) continue;
          const event = JSON.parse(line.slice(6));
          if (event.type === "status") setStatusMessage(event.message);
          if (event.type === "tool") {
            const count = event.data.result_count;
            setStatusMessage(`${event.data.tool} 已完成${typeof count === "number" ? ` · ${count} 条结果` : ""}`);
          }
          if (event.type === "result") {
            setCurrentConversationId(event.data.conversation_id);
            setMessages((items) => [...items, {
              id: event.data.message_id || crypto.randomUUID(),
              role: "assistant",
              content: event.data.answer,
              citations: event.data.citations || [],
              memorySuggestion: event.data.memory_suggestion,
              plan: event.data.plan || null,
            }]);
            void searchConversations(conversationQuery);
            void loadRuntime();
          }
          if (event.type === "error") throw new Error(event.message);
          if (event.type === "done") setStatusMessage("");
        }
        if (done) break;
      }
      setStatusMessage("");
    } catch (error) {
      setStatusMessage("");
      setMessages((items) => [...items, {
        id: crypto.randomUUID(),
        role: "assistant",
        content: `这次没有完成：${error instanceof Error ? error.message : "未知错误"}。请确认本地后端和模型连接状态。`,
      }]);
    }
  }

  async function updateMemory(memoryId: string, patch: { content?: string; status?: string }) {
    await api(`/api/memories/${memoryId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    setMessages((items) => items.map((item) => item.memorySuggestion?.id === memoryId
      ? { ...item, memorySuggestion: { ...item.memorySuggestion, status: patch.status || item.memorySuggestion.status, content: patch.content || item.memorySuggestion.content } }
      : item));
    await loadRuntime();
  }

  async function regenerateAssistant(messageId: string) {
    if (!currentConversationId || statusMessage) return;
    setStatusMessage("正在重新执行上一条问题");
    setSource(null);
    try {
      const result = await api<{
        message_id: string; answer: string; citations: Citation[];
        memory_suggestion?: MemorySuggestion | null; plan?: AgentPlan | null;
      }>("/api/chat/regenerate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: currentConversationId,
          assistant_message_id: messageId,
        }),
      });
      setMessages((items) => items.map((item) => item.id === messageId ? {
        ...item,
        content: result.answer,
        citations: result.citations || [],
        memorySuggestion: result.memory_suggestion,
        plan: result.plan || null,
      } : item));
      void searchConversations(conversationQuery);
      void loadRuntime();
    } catch (error) {
      window.alert(error instanceof Error ? error.message : "重新生成失败");
    } finally {
      setStatusMessage("");
    }
  }

  async function createMemory(content: string) {
    await api("/api/memories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content, kind: "preference" }),
    });
    await loadRuntime();
  }

  async function deleteMemory(memoryId: string) {
    if (!window.confirm("永久删除这条 Memory？")) return;
    await api(`/api/memories/${memoryId}`, { method: "DELETE" });
    await loadRuntime();
  }

  async function showDocumentStatus(documentId: string) {
    setSelectedDocumentStatus(await api<DocumentStatus>(`/api/documents/${documentId}/status`));
  }

  async function reindexDocument(documentId: string) {
    await api(`/api/documents/${documentId}/reindex`, { method: "POST" });
    setSelectedDocumentStatus(await api<DocumentStatus>(`/api/documents/${documentId}/status`));
    await loadRuntime();
  }

  async function createSpace(name: string) {
    const created = await api<Space>("/api/spaces", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    await loadRuntime();
    chooseSpace(created.id);
    return created;
  }

  async function openCitation(citation: Citation) {
    if (citation.kind === "web" && citation.url) {
      window.open(citation.url, "_blank", "noopener,noreferrer");
      return;
    }
    setSource(citation);
    try {
      const detail = await api<{ text: string; heading: string }>(`/api/chunks/${citation.chunk_id}`);
      setSource({ ...citation, ...detail });
    } catch {
      setSource(citation);
    }
  }

  function finishOnboarding() {
    localStorage.setItem("kun.onboarding.complete", "1");
    setOnboarding(false);
  }

  return (
    <main className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "" : "collapsed"}`}>
        <div className="brand-row">
          <div className="brand-mark">K</div>
          {sidebarOpen && <div><strong>KUN</strong><span>Personal Knowledge Agent</span></div>}
          <button className="icon-button sidebar-toggle" onClick={() => setSidebarOpen(!sidebarOpen)} aria-label="收起侧栏">{sidebarOpen ? "‹" : "›"}</button>
        </div>
        <button className="new-chat" onClick={newChat}><span>＋</span>{sidebarOpen && "新对话"}</button>
        <nav>
          {NAV.map((item) => <button key={item.id} className={view === item.id ? "active" : ""} onClick={() => setView(item.id)}><span>{item.icon}</span>{sidebarOpen && item.label}</button>)}
        </nav>
        {sidebarOpen && <section className="history"><p>最近对话</p><div className="history-search"><span>⌕</span><input value={conversationQuery} onChange={(event) => void searchConversations(event.target.value)} placeholder="搜索对话" /></div>{conversations.length ? <div className="history-list">{conversations.map((conversation) => <div className={conversation.id === currentConversationId ? "current" : ""} key={conversation.id}><button onClick={() => void openConversation(conversation.id)} title={conversation.preview || conversation.title}>{conversation.title}</button><button className="history-delete" onClick={() => void deleteConversation(conversation.id)} aria-label={`删除对话 ${conversation.title}`}>×</button></div>)}</div> : <div className="history-empty">{conversationQuery ? "没有匹配的对话" : "发送第一条消息后会自动保存在本机"}</div>}</section>}
        <div className="sidebar-footer">
          <button className={view === "settings" ? "active" : ""} onClick={() => setView("settings")}><span>⚙</span>{sidebarOpen && "设置"}</button>
          {sidebarOpen && <div className={`local-state ${health ? "" : "offline"}`}><i />{health ? "本地服务正常" : "本地服务未连接"}<span>{documents.length ? `${documents.length} 份资料` : "资料库为空"}</span></div>}
        </div>
      </aside>

      <section className="workspace">
        {view !== "chat" && <header className="topbar">
          <div className="space-selector"><span className="space-dot" />{viewTitle}</div>
          <div className="top-actions">
            <span className="privacy-pill">⌂ 本地优先</span>
            <button className="avatar">坤</button>
          </div>
        </header>}

        {view === "learning" && <LearningView navigate={setView} />}
        {view === "chat" && <ChatView messages={messages} statusMessage={statusMessage} source={source} setSource={setSource} openCitation={openCitation} message={message} setMessage={setMessage} submitMessage={submitMessage} regenerateAssistant={regenerateAssistant} openFiles={() => fileInput.current?.click()} pasteFiles={pasteFiles} activeSpace={activeSpace} spaces={spaces} chooseSpace={chooseSpace} updateMemory={updateMemory} />}
        {view === "spaces" && <SpacesView spaces={spaces} activeSpaceId={activeSpace?.id} selectSpace={(id) => { chooseSpace(id); setView("chat"); }} createSpace={createSpace} />}
        {view === "library" && <LibraryView documents={documents} spaces={spaces} openFiles={() => fileInput.current?.click()} showStatus={showDocumentStatus} />}
        {view === "images" && <ImageSearchView activeSpace={activeSpace} openFiles={() => fileInput.current?.click()} />}
        {view === "skills" && <SkillsView />}
        {view === "tools" && <ToolsView />}
        {view === "memory" && <MemoryView memories={memories} currentConversationId={currentConversationId} shortTermRevision={messages.length} updateMemory={updateMemory} createMemory={createMemory} deleteMemory={deleteMemory} />}
        {view === "lab" && <RagLabView activeSpace={activeSpace} documents={documents.filter((item) => item.space_id === activeSpace?.id)} />}
        {view === "infra" && <InfraView activeSpace={activeSpace} health={health} />}
        {view === "settings" && <SettingsView providers={providers} health={health} testProvider={testProvider} />}
      </section>

      <input ref={fileInput} type="file" multiple hidden accept=".pdf,.docx,.md,.xlsx,.xls,.png,.jpg,.jpeg" onChange={chooseFiles} />
      {uploadOpen && <UploadReview staged={staged} imported={imported} status={uploadStatus} progress={uploadProgress} error={uploadError} close={() => setUploadOpen(false)} confirm={confirmImport} update={updateStaged} />}
      {selectedDocumentStatus && <IndexStatusModal status={selectedDocumentStatus} close={() => setSelectedDocumentStatus(null)} reindex={() => void reindexDocument(selectedDocumentStatus.id)} />}
      {onboarding && <Onboarding step={onboardingStep} setStep={setOnboardingStep} finish={finishOnboarding} providers={providers} testProvider={testProvider} backendOnline={Boolean(health)} />}
    </main>
  );
}

function ChatView(props: {
  messages: ChatMessage[];
  statusMessage: string;
  source: SourcePreview | null;
  setSource: (value: SourcePreview | null) => void;
  openCitation: (citation: Citation) => void;
  message: string;
  setMessage: (value: string) => void;
  submitMessage: () => void;
  regenerateAssistant: (messageId: string) => Promise<void>;
  openFiles: () => void;
  pasteFiles: (files: File[]) => void;
  updateMemory: (memoryId: string, patch: { content?: string; status?: string }) => Promise<void>;
  activeSpace?: Space;
  spaces: Space[];
  chooseSpace: (spaceId: string) => void;
}) {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [activeQuestionId, setActiveQuestionId] = useState<string | null>(null);
  const messageScrollRef = useRef<HTMLDivElement>(null);
  const [sourceWidth, setSourceWidth] = useState(() => {
    const saved = Number(localStorage.getItem("kun.sourcePanel.width"));
    return Number.isFinite(saved) && saved >= 320 ? saved : 430;
  });
  const questions = useMemo(() => props.messages.filter((item) => item.role === "user"), [props.messages]);

  useEffect(() => {
    if (!questions.length) {
      setActiveQuestionId(null);
      return;
    }
    setActiveQuestionId((current) => current && questions.some((item) => item.id === current) ? current : questions[questions.length - 1].id);
  }, [questions]);

  function updateActiveQuestion() {
    const scroller = messageScrollRef.current;
    if (!scroller) return;
    const rows = Array.from(scroller.querySelectorAll<HTMLElement>("[data-question-id]"));
    const threshold = scroller.getBoundingClientRect().top + 130;
    let active = rows[0]?.dataset.questionId || null;
    let nearestDistance = Number.POSITIVE_INFINITY;
    for (const row of rows) {
      const distance = Math.abs(row.getBoundingClientRect().top - threshold);
      if (distance < nearestDistance) {
        nearestDistance = distance;
        active = row.dataset.questionId || active;
      }
    }
    if (active) setActiveQuestionId(active);
  }

  function jumpToQuestion(id: string) {
    document.getElementById(`question-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    setActiveQuestionId(id);
  }

  function boundedSourceWidth(value: number) {
    const available = Math.max(320, window.innerWidth - 720);
    return Math.round(Math.min(Math.min(900, available), Math.max(320, value)));
  }

  function startSourceResize(event: ReactPointerEvent<HTMLDivElement>) {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = sourceWidth;
    let latestWidth = sourceWidth;
    document.body.classList.add("resizing-source");
    const move = (moveEvent: PointerEvent) => {
      latestWidth = boundedSourceWidth(startWidth + startX - moveEvent.clientX);
      setSourceWidth(latestWidth);
    };
    const finish = () => {
      document.body.classList.remove("resizing-source");
      localStorage.setItem("kun.sourcePanel.width", String(latestWidth));
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", finish);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", finish, { once: true });
  }

  function resizeSourceWithKeyboard(delta: number) {
    const next = boundedSourceWidth(sourceWidth + delta);
    setSourceWidth(next);
    localStorage.setItem("kun.sourcePanel.width", String(next));
  }

  function resetSourceWidth() {
    const next = boundedSourceWidth(430);
    setSourceWidth(next);
    localStorage.setItem("kun.sourcePanel.width", String(next));
  }

  function importFromClipboard(event: ClipboardEvent<HTMLTextAreaElement>) {
    const files = Array.from(event.clipboardData.files);
    if (!files.length) return;
    event.preventDefault();
    props.pasteFiles(files);
  }
  function importFromDrop(event: DragEvent<HTMLDivElement>) {
    const files = Array.from(event.dataTransfer.files);
    if (!files.length) return;
    event.preventDefault();
    props.pasteFiles(files);
  }
  async function copyMessage(id: string, content: string) {
    try {
      let copied = false;
      if (navigator.clipboard?.writeText) {
        try {
          await navigator.clipboard.writeText(content);
          copied = true;
        } catch {
          // Fall through for WebView or browsers that deny Clipboard API.
        }
      }
      if (!copied) {
        const textarea = document.createElement("textarea");
        textarea.value = content;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.appendChild(textarea);
        textarea.select();
        copied = document.execCommand("copy");
        textarea.remove();
      }
      if (!copied) throw new Error("浏览器未允许读取剪贴板");
      setCopiedId(id);
      window.setTimeout(() => setCopiedId((current) => current === id ? null : current), 1400);
    } catch {
      window.alert("复制失败，请选中文本后按 Ctrl+C。");
    }
  }
  const lastAssistantId = [...props.messages].reverse().find((item) => item.role === "assistant")?.id;
  return <div className={`chat-layout ${props.source ? "with-source" : ""}`}>
    <section className="conversation">
      <header className="chat-topbar">
        <span className="privacy-pill">⌂ 本地优先</span>
        <button className="avatar">坤</button>
        <div className="chat-space-selector"><span className="space-dot" /><select value={props.activeSpace?.id || ""} onChange={(event) => props.chooseSpace(event.target.value)} aria-label="当前知识空间">{props.spaces.map((space) => <option value={space.id} key={space.id}>{space.name}</option>)}</select></div>
      </header>
      <div className="messages-scroll" ref={messageScrollRef} onScroll={updateActiveQuestion}>
      <div className={`messages ${props.messages.length ? "" : "empty-conversation"}`}>
        {!props.messages.length && <div className="chat-welcome"><div className="kun-orb">K</div><h1>从你的资料开始</h1><p>添加文件并确认建立索引后，坤坤会基于真实内容回答，并标出可核对的来源。</p><button className="primary-button" onClick={props.openFiles}>＋ 添加第一份资料</button></div>}
        {props.messages.map((item) => item.role === "user"
          ? <div className="user-row" id={`question-${item.id}`} data-question-id={item.id} key={item.id}><div className="user-message-wrap"><div className="user-bubble">{item.content}</div><div className="message-actions user-actions"><button onClick={() => void copyMessage(item.id, item.content)} title="复制">{copiedId === item.id ? "✓ 已复制" : "▣ 复制"}</button></div></div></div>
          : <div className="agent-message" key={item.id}><div className="kun-orb small">K</div><div className="answer-body"><AnswerText message={item} openCitation={props.openCitation} updateMemory={props.updateMemory} /><div className="message-actions assistant-actions"><button onClick={() => void copyMessage(item.id, item.content)} title="复制回答">{copiedId === item.id ? "✓ 已复制" : "▣ 复制"}</button>{item.id === lastAssistantId && <button onClick={() => void props.regenerateAssistant(item.id)} disabled={Boolean(props.statusMessage)} title="重新生成">↻ 重新生成</button>}</div></div></div>
        )}
        {props.statusMessage && <div className="agent-message thinking"><div className="kun-orb small">K</div><div><strong>坤坤正在处理</strong><p><span className="pulse-dot" />{props.statusMessage}</p></div></div>}
      </div>
      </div>
      {questions.length > 1 && <nav className="question-rail" aria-label="快速跳转到本次对话中的问题">
        {questions.map((item, index) => <button
          type="button"
          key={item.id}
          className={activeQuestionId === item.id ? "active" : ""}
          aria-label={`跳转到第 ${index + 1} 个问题：${item.content}`}
          data-label={`${index + 1}. ${item.content.replace(/\s+/g, " ").slice(0, 32)}${item.content.length > 32 ? "…" : ""}`}
          onClick={() => jumpToQuestion(item.id)}
        ><span>{index + 1}</span></button>)}
      </nav>}
      <div className="composer-wrap">
        <div className="composer" onDragOver={(event) => event.preventDefault()} onDrop={importFromDrop}>
          <textarea value={props.message} onChange={(event) => props.setMessage(event.target.value)} onPaste={importFromClipboard} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); props.submitMessage(); } }} placeholder="问问坤坤，粘贴图片或文件…" rows={1} />
          <div className="composer-actions"><div><button onClick={props.openFiles} aria-label="添加资料">＋</button><span className="context-button">◫ {props.activeSpace?.name || "默认知识空间"}</span></div><button className="send-button" onClick={props.submitMessage} disabled={Boolean(props.statusMessage)}>↑</button></div>
        </div>
        <p className="disclaimer">坤坤可能会犯错，重要信息请核对引用来源。</p>
      </div>
    </section>
    {props.source && <>
      <div
        className="source-resizer"
        role="separator"
        aria-label="调整聊天与文档区域宽度"
        aria-orientation="vertical"
        aria-valuemin={320}
        aria-valuemax={900}
        aria-valuenow={sourceWidth}
        tabIndex={0}
        title="拖动调整文档宽度，双击恢复默认"
        onPointerDown={startSourceResize}
        onDoubleClick={resetSourceWidth}
        onKeyDown={(event) => {
          if (event.key === "ArrowLeft") resizeSourceWithKeyboard(24);
          if (event.key === "ArrowRight") resizeSourceWithKeyboard(-24);
        }}
      ><span /></div>
      <SourcePanel source={props.source} width={sourceWidth} close={() => props.setSource(null)} />
    </>}
  </div>;
}

function PlanFlow({ plan }: { plan: AgentPlan }) {
  const statusLabel: Record<PlanTaskStatus, string> = {
    pending: "等待执行", in_progress: "执行中", completed: "已完成",
    awaiting_confirmation: "等你确认", failed: "未通过",
  };
  const intentLabel: Record<string, string> = {
    web_research: "联网研究", knowledge_question: "本地知识问答", memory_query: "Memory 查询",
    memory_setting: "Memory 整理", image_search: "图片搜索", table_analysis: "表格分析", video_learning: "视频学习",
  };
  const inferredSkill: Record<string, string> = { web_research: "web_research_skill", knowledge_question: "document_skill", memory_query: "memory_skill", memory_setting: "memory_skill", image_search: "image_skill", table_analysis: "excel_skill", video_learning: "video_skill" };
  const selectedSkill = plan.skill || inferredSkill[plan.intent] || "旧版记录未保存";
  const risk = plan.grounding?.risk || "unknown";
  return <details className="agent-plan" open>
    <summary><span className="plan-symbol">⌁</span><div><strong>Agent 执行解释</strong><small>{plan.tasks.length} 个步骤 · {intentLabel[plan.intent] || plan.intent} · {plan.status === "completed" ? "本轮完成" : plan.status ? "正在执行" : "执行前预览"}</small></div><b>收起 / 展开</b></summary>
    <div className="plan-route"><span>用户目标</span><strong>{plan.goal}</strong><i>→</i><span>选择 Skill</span><strong>{selectedSkill}</strong><p>{plan.route_reason || "根据请求类型选择最小必要能力。"}</p>
      {plan.fallback_skill && <div className="plan-fallback"><b>本地未命中</b><i>→</i><span>切换 Skill</span><strong>{plan.fallback_skill}</strong><p>{plan.fallback_reason || "本地候选与问题不相关，自动改用公开网页证据。"}</p></div>}
    </div>
    <ol>{plan.tasks.map((task, index) => <li className={`plan-${task.status}`} key={task.id}>
      <i>{task.status === "completed" ? "✓" : task.status === "awaiting_confirmation" ? "!" : task.status === "failed" ? "×" : String(index + 1)}</i>
      <div><strong>{task.title}</strong><small>{task.detail || task.source}</small></div>
      <span>{statusLabel[task.status]}</span>
    </li>)}</ol>
    {!!plan.tool_calls?.length && <div className="plan-tools"><b>真实 Tool 调用</b>{plan.tool_calls.map((tool, index) => <span key={`${tool.name}-${index}`}><code>{tool.name}</code><i>{tool.status === "succeeded" ? "成功" : tool.recoverable ? "失败 · 已降级" : tool.status || "未知"} · {tool.result_count ?? 0} 条 · {tool.duration_ms ?? 0} ms{tool.error_code ? ` · ${tool.error_code}` : ""}</i></span>)}</div>}
    {plan.grounding && <div className={`grounding-card risk-${risk}`}><span>幻觉风险</span><strong>{plan.grounding.label}</strong><b>{risk === "low" ? "较低" : risk === "medium" ? "中等" : risk === "high" ? "较高" : "评估中"}</b><p>{plan.grounding.explanation}</p></div>}
    <p className="plan-explainer">{plan.explanation || "Plan 决定做什么；Skill 规定怎么做；Tool 返回可观察结果；模型根据证据组织语言。"}</p>
  </details>;
}
function AnswerText({ message, openCitation, updateMemory }: {
  message: ChatMessage;
  openCitation: (citation: Citation) => void;
  updateMemory: (memoryId: string, patch: { content?: string; status?: string }) => Promise<void>;
}) {
  const citations = new Map((message.citations || []).map((item) => [item.id, item]));
  const markdown = message.content.replace(/\[(\d+)\]/g, (whole, rawId) => {
    const id = Number(rawId);
    return citations.has(id) ? `[[${id}]](#citation-${id})` : whole;
  });
  return <>
    {message.plan?.tasks?.length ? <PlanFlow plan={message.plan} /> : null}
    <div className="markdown-answer"><ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ href, children }) => {
          const match = href?.match(/^#citation-(\d+)$/);
          const citation = match ? citations.get(Number(match[1])) : undefined;
          return citation
            ? <button className={`citation ${citation.kind === "web" ? "web-citation" : ""}`} onClick={() => openCitation(citation)} title={citation.kind === "web" ? `${citation.title} · 打开网页` : `${citation.file} · ${citation.locator}`}>{children}</button>
            : <a href={href} target="_blank" rel="noreferrer">{children}</a>;
        },
      }}
    >{markdown}</ReactMarkdown></div>
    {message.memorySuggestion && message.memorySuggestion.status === "pending" && <div className="memory-suggestion"><span>◇</span><div><strong>要让坤坤长期记住吗？</strong><p>{message.memorySuggestion.content}</p></div><button onClick={() => void updateMemory(message.memorySuggestion!.id, { status: "dismissed" })}>忽略</button><button className="memory-accept" onClick={() => void updateMemory(message.memorySuggestion!.id, { status: "enabled" })}>记住</button></div>}
    {message.memorySuggestion?.status === "enabled" && <div className="memory-saved">✓ 已保存到长期 Memory，可随时在 Memory 中心修改</div>}
    {!!message.citations?.length && <div className={`source-strip ${message.citations.some((item) => item.kind === "web") ? "web-source-strip" : ""}`} onClick={() => openCitation(message.citations![0])}><span className="stacked-files">{message.citations.some((item) => item.kind === "web") ? "WEB" : "SRC"}</span><div><strong>{message.citations.some((item) => item.kind === "web") ? `搜索并引用了 ${message.citations.length} 个网页` : `${message.citations.length} 处有效引用`}</strong><small>{message.citations.map((item) => item.kind === "web" ? (item.site_name || item.file) : `${item.file} · ${item.locator}`).join("　")}</small></div><b>{message.citations.some((item) => item.kind === "web") ? "打开网页 ›" : "查看来源 ›"}</b></div>}
  </>;
}

function SourcePanel({ source, width, close }: { source: SourcePreview; width: number; close: () => void }) {
  return <aside className="source-panel" style={{ width }}>
    <div className="source-head"><div><span>{source.file.split(".").pop()?.toUpperCase()}</span><strong>{source.file}</strong></div><button onClick={close} aria-label="关闭来源">×</button></div>
    <div className="source-toolbar"><span>{source.locator}</span></div>
    <div className="document-preview"><div className="paper"><small>{source.locator}</small><h2>{source.heading || source.title}</h2><p>{source.text || "正在读取引用片段…"}</p></div></div>
    <div className="match-note"><span>✓</span><div><strong>回答引用位置</strong><p>{source.locator} · Chunk {source.chunk_id.slice(0, 8)}</p></div></div>
  </aside>;
}

function PageHead({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: ReactNode }) {
  return <div className="page-head"><div><span>{eyebrow}</span><h1>{title}</h1><p>{description}</p></div>{action}</div>;
}

function SpacesView({ spaces, activeSpaceId, selectSpace, createSpace }: {
  spaces: Space[];
  activeSpaceId?: string;
  selectSpace: (id: string) => void;
  createSpace: (name: string) => Promise<Space>;
}) {
  const [name, setName] = useState("");
  const [creating, setCreating] = useState(false);
  async function submit() {
    if (!name.trim()) return;
    setCreating(true);
    try {
      const created = await createSpace(name.trim());
      setName("");
      selectSpace(created.id);
    } finally {
      setCreating(false);
    }
  }
  return <div className="page">
    <PageHead eyebrow="KNOWLEDGE SPACES" title="知识空间" description="资料、对话上下文和检索范围按主题隔离，切换后新对话只查询当前空间。" action={<div className="space-create"><input value={name} onChange={(event) => setName(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void submit(); }} placeholder="新空间名称" /><button className="primary-button" disabled={creating || !name.trim()} onClick={() => void submit()}>＋ 创建</button></div>} />
    {!spaces.length ? <EmptyState title="还没有知识空间" text="本地后端启动后会创建默认知识空间。" /> : <div className="space-grid">{spaces.map((space) => <button className={`space-card ${space.id === activeSpaceId ? "selected-space" : ""}`} key={space.id} onClick={() => selectSpace(space.id)}><div className="space-icon" style={{ background: space.color }}>K</div><h3>{space.name}</h3><p>{space.document_count} 份资料 · {space.chunk_count} 个知识片段</p><div className="health"><i /><span>{space.chunk_count ? "索引可检索" : "等待添加资料"}</span><small>{space.id === activeSpaceId ? "当前空间" : "进入空间 ›"}</small></div></button>)}</div>}
  </div>;
}

function LearningView({ navigate }: { navigate: (view: View) => void }) {
  const [overview, setOverview] = useState<LearningOverview | null>(null);
  const [traces, setTraces] = useState<AgentTrace[]>([]);
  const [error, setError] = useState("");
  useEffect(() => {
    Promise.all([api<LearningOverview>("/api/learning/overview"), api<AgentTrace[]>("/api/agent/traces?limit=12")])
      .then(([nextOverview, nextTraces]) => { setOverview(nextOverview); setTraces(nextTraces); })
      .catch((reason) => setError(reason instanceof Error ? reason.message : "学习概览读取失败"));
  }, []);
  const statusLabel: Record<CapabilityStatus, string> = { verified: "本机已验证", partial: "已实现 · 待运行验证", planned: "待补充" };
  if (error) return <div className="page"><PageHead eyebrow="AGENT INSIGHT" title="Agent 透视" description="用真实运行数据解释 KUN 如何工作。" /><div className="lab-error">{error}</div></div>;
  if (!overview) return <div className="page learning-page"><PageHead eyebrow="AGENT INSIGHT" title="Agent 透视" description="正在读取本机能力状态…" /></div>;
  return <div className="page learning-page">
    <PageHead eyebrow="AGENT INSIGHT" title="Agent 透视" description="看懂一份资料如何变成可检索证据、带引用回答和可控 Memory；状态来自本机数据库与能力配置。" />
    <section className="learning-stats">
      <article><span>本地资料</span><strong>{overview.documents.total}</strong><small>{overview.documents.ready} 份可检索 · {overview.documents.chunks} 个 Chunk</small></article>
      <article><span>长期 Memory</span><strong>{overview.memories.enabled}</strong><small>{overview.memories.pending} 条待确认 · {overview.memories.events} 条事件</small></article>
      <article><span>Tool Runtime</span><strong>{overview.tools.available.length}/{overview.tools.all.length}</strong><small>{overview.tools.succeeded || 0} 次成功 · {overview.tools.failed || 0} 次失败</small></article>
      <article><span>真实评估</span><strong>{overview.evaluation.case_count}</strong><small>{overview.evaluation.latest ? "已有最近运行结果" : "尚未运行，不显示虚假分数"}</small></article>
    </section>
    <section className="learning-block"><header><div><span>从资料到可控记忆</span><h2>一条完整 Agent 链路</h2></div><p>点任一步可进入真实功能页面；“本机已验证”只表示当前配置生命周期里已有运行证据。</p></header>
      <div className="learning-flow">{overview.workflow.map((step, index) => <button key={step.id} onClick={() => navigate(step.route)}><i>{index + 1}</i><span><b>{step.label}</b><small>{step.plain}</small><em className={`cap-status ${step.status}`}>{statusLabel[step.status]}</em></span>{index < overview.workflow.length - 1 && <strong>→</strong>}</button>)}</div>
    </section>
    <section className="context-budget learning-block"><header><div><span>CONTEXT BUDGET</span><h2>本轮上下文是怎样装进模型的</h2></div><button className="secondary-button" onClick={() => navigate("memory")}>查看当前对话构成</button></header>
      <div className="budget-strip"><div><span>模型硬上限</span><strong>{overview.context_policy.model_hard_limit.toLocaleString()}</strong></div><i>−</i><div><span>安全余量</span><strong>{overview.context_policy.safety_margin.toLocaleString()}</strong></div><i>−</i><div><span>输出预留</span><strong>{overview.context_policy.output_reserve.toLocaleString()}</strong></div><i>→</i><div className="budget-result"><span>有效输入上限</span><strong>{overview.context_policy.effective_input_limit.toLocaleString()}</strong></div></div>
      <p>完整原始对话始终保留在本机。模型输入使用“结构化历史摘要 + 预算内最近完整消息”；当前数字是可配置工程参数，Token 为确定性估算，不冒充模型 tokenizer 精确值。</p>
    </section>
    <section className="learning-block capability-section"><header><div><span>CAPABILITY MATRIX</span><h2>能力矩阵</h2></div><p>每项都给出本机证据、验证入口和仍需补齐的边界。</p></header>
      <div className="capability-table"><div className="capability-row capability-head"><span>面试考点</span><span>当前能力</span><span>真实状态</span><span>如何验证</span><span>下一步</span></div>{overview.capability_matrix.map((item) => <div className="capability-row" key={item.id}><strong>{item.topic}</strong><span>{item.capability}<small>{item.evidence}</small></span><em className={`cap-status ${item.status}`}>{statusLabel[item.status]}</em><button onClick={() => navigate(item.route)}>{item.verify} ›</button><span>{item.next}</span></div>)}</div>
    </section>
    <section className="learning-block trace-section"><header><div><span>AGENT TRACE</span><h2>最近真实运行</h2></div><p>只展示可审计阶段、耗时和结果摘要，不展示模型私密思维链。</p></header>
      {traces.length ? <div className="trace-list">{traces.map((trace) => <details key={trace.id}><summary><i className={trace.status} /><span><strong>{trace.intent || "未识别意图"}</strong><small>{trace.selected_skill || "未选择 Skill"} · {trace.duration_ms ?? 0} ms · {formatDate(trace.started_at)}</small></span><b>{trace.retrieval_count} 条检索 · {trace.citation_count} 个引用</b></summary><div className="trace-meta"><span>上下文估算 {trace.context_tokens}/{trace.context_budget}</span><span>摘要版本 {trace.summary_version ?? "未生成"}</span><span>路由目录 {trace.exposed_tool_count} 个 Tool</span><span>已选 Schema 估算 {trace.schema_token_estimate} Token</span></div><ol>{trace.stages.map((stage) => <li key={stage.id}><i /><div><strong>{stage.stage_name}</strong><small>{stage.status} · {stage.duration_ms} ms</small></div><code>{JSON.stringify(stage.result_summary)}</code></li>)}</ol></details>)}</div> : <div className="lab-start"><b>还没有 Agent Trace</b><p>发送一次问题后，这里会出现真实阶段、上下文预算、检索数量和引用数量。</p><button className="primary-button" onClick={() => navigate("chat")}>去发起一次对话</button></div>}
    </section>
  </div>;
}
function LibraryView({ documents, spaces, openFiles, showStatus }: { documents: DocumentItem[]; spaces: Space[]; openFiles: () => void; showStatus: (id: string) => Promise<void> }) {
  const names = new Map(spaces.map((item) => [item.id, item.name]));
  const statusLabels: Record<string, string> = {
    ready: "可检索",
    lexical_ready: "关键词可检索 · 语义索引中",
    indexing: "正在索引",
    failed: "索引失败",
    missing: "本地副本缺失",
    stale: "待重新索引",
  };
  return <div className="page">
    <PageHead eyebrow="LOCAL LIBRARY" title="文件资料库" description="KUN 保存独立副本；删除原位置文件不会影响这里的资料和索引。" action={<button className="primary-button" onClick={openFiles}>＋ 添加资料</button>} />
    {!documents.length ? <EmptyState title="资料库还是空的" text="上传文件后，确认 Agent 生成的标题和摘要，再建立索引。" action={<button className="primary-button" onClick={openFiles}>选择文件</button>} /> : <div className="table">
      <div className="table-row table-head"><span>文件</span><span>知识空间</span><span>类型</span><span>索引状态</span><span>更新</span></div>
      {documents.map((document) => { const status = document.effective_index_status || document.index_status; return <div className="table-row" key={document.id}><span className="file-cell"><b className={`file-badge ${document.file_type.slice(0, 2)}`}>{document.file_type.slice(0, 2).toUpperCase()}</b><span><strong>{document.title}</strong><small>{document.original_name}</small></span></span><span>{names.get(document.space_id) || document.space_id}</span><span>{document.file_type.toUpperCase()}</span><button className={`index-status-button status-${status}`} onClick={() => void showStatus(document.id)}><i className="ok-dot" />{statusLabels[status] || status}<b>›</b></button><span>{formatDate(document.updated_at)}</span></div>; })}
    </div>}
  </div>;
}

function MemoryView({ memories, currentConversationId, shortTermRevision, updateMemory, createMemory, deleteMemory }: {
  memories: MemoryItem[];
  currentConversationId: string | null;
  shortTermRevision: number;
  updateMemory: (id: string, patch: { content?: string; status?: string }) => Promise<void>;
  createMemory: (content: string) => Promise<void>;
  deleteMemory: (id: string) => Promise<void>;
}) {
  const [draft, setDraft] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editingText, setEditingText] = useState("");
  const [shortTerm, setShortTerm] = useState<ShortTermMemory | null>(null);
  const [audit, setAudit] = useState<MemoryDetail | null>(null);
  useEffect(() => {
    const query = currentConversationId ? `?conversation_id=${encodeURIComponent(currentConversationId)}` : "";
    void api<ShortTermMemory>(`/api/memories/short-term${query}`).then(setShortTerm).catch(() => setShortTerm(null));
  }, [currentConversationId, memories, shortTermRevision]);
  const pending = memories.filter((item) => item.status === "pending");
  const saved = memories.filter((item) => item.status !== "pending");
  return <div className="page">
    <PageHead eyebrow="CONTROLLED MEMORY" title="Memory 中心" description="短期 Memory 服务当前对话；长期 Memory 保存稳定背景，并始终经过你的确认。" action={<div className="memory-create"><input value={draft} onChange={(event) => setDraft(event.target.value)} placeholder="也可以手动添加稳定偏好" /><button className="primary-button" disabled={!draft.trim()} onClick={() => { void createMemory(draft.trim()); setDraft(""); }}>添加记忆</button></div>} />
    {audit && <section className="memory-audit"><header><div><span>MEMORY AUDIT</span><h2>{audit.content}</h2><p>版本 v{audit.version || 1} · 来源 {audit.source}{audit.conflict_with_id ? " · 检测到潜在冲突" : ""}</p></div><button onClick={() => setAudit(null)}>关闭</button></header><div><section><h3>变更事件</h3>{audit.events.length ? audit.events.map((event) => <p key={event.id}><b>{event.event_type}</b><span>v{event.version} · {event.source} · {formatDate(event.created_at)}</span></p>) : <small>旧数据尚无事件记录</small>}</section><section><h3>为什么召回</h3>{audit.recent_recalls.length ? audit.recent_recalls.map((recall) => <p key={recall.id}><b>{recall.reason}</b><span>{recall.query} · {formatDate(recall.created_at)}</span></p>) : <small>这条 Memory 还没有被按需召回</small>}</section></div></section>}
    <section className="memory-section short-term-memory"><h2>短期 Memory <span>当前对话窗口</span></h2>
      {shortTerm?.conversation ? <article className="short-term-dashboard">
        <div className="short-memory-stats">
          <div><span>模型窗口</span><strong>{shortTerm.window_size}</strong><small>条消息</small></div>
          <div><span>本地对话</span><strong>{shortTerm.message_count}</strong><small>条已保存</small></div>
          <div><span>事实线索</span><strong>{shortTerm.working_facts?.length || 0}</strong><small>条已识别</small></div>
        </div>
        <div className="short-memory-explainer"><span>窗口如何工作</span><b>最多 {shortTerm.previous_message_limit} 条上一轮消息</b><i>＋</i><b>当前问题</b><i>→</i><b>生成本次回答</b><p>{shortTerm.storage_policy}。窗口外的消息没有丢失，重新打开历史对话仍能看到。</p></div>
        <div className="short-memory-grid">
          <section className="short-term-summary"><span>当前工作上下文</span><h3>{shortTerm.conversation.title}</h3><p>{shortTerm.expires}；稳定信息只有经过确认才会进入长期 Memory。</p>{shortTerm.working_facts?.length ? <div className="working-facts">{shortTerm.working_facts.map((fact) => <b key={`${fact.kind}-${fact.content}`} className={`fact-${fact.status}`}><span>{fact.content}</span><i>{fact.status === "enabled" ? "已进入长期" : fact.status === "pending" ? "待确认" : "仅当前对话"}</i></b>)}</div> : <div className="facts-empty">最近窗口中还没有识别到姓名、地点、偏好或目标等稳定信息。</div>}</section>
          <section className="short-message-panel"><header><div><span>最近消息时间线</span><strong>当前显示 {shortTerm.messages.length} / {shortTerm.window_size} 条</strong></div><small>越靠下越新</small></header><div className="short-message-list">{shortTerm.messages.map((item, index) => <div key={item.id}><span className={item.role}>{item.role === "user" ? "你" : "K"}</span><i /><div><b>{item.role === "user" ? "你的问题" : "坤坤的回答"}</b><p>{item.content}</p></div><small>{index + 1}</small></div>)}</div></section>
        </div>
      </article>
        : <div className="short-term-empty">开始一段对话后，这里会显示当前工作记忆；它不会自动变成长期记忆。</div>}
    </section>
    {!!pending.length && <section className="memory-section"><h2>等待你确认 <span>{pending.length}</span></h2><div className="memory-list">{pending.map((item) => <article className="pending-memory" key={item.id}><div className="memory-symbol">?</div><div><span>来自对话的建议</span><h3>{item.content}</h3><p>{formatDate(item.updated_at)} · 未确认前不会参与回答</p></div><div className="memory-actions"><button onClick={() => void updateMemory(item.id, { status: "dismissed" })}>忽略</button><button className="secondary-button" onClick={() => void updateMemory(item.id, { status: "enabled" })}>确认记住</button></div></article>)}</div></section>}
    <section className="memory-section"><h2>长期 Memory <span>{saved.length}</span></h2>{!saved.length ? <EmptyState title="还没有已确认的长期记忆" text="对话中的姓名、所在地、稳定偏好、目标和项目背景会先形成候选；确认后才保存。" /> : <div className="memory-list">{saved.map((item) => <article key={item.id}><div className="memory-symbol">◇</div><div><span>{{ preference: "表达与学习偏好", identity: "身份信息", location: "所在地", goal: "长期目标", project: "当前项目", relationship: "人物关系" }[item.kind] || item.kind}</span>{editingId === item.id ? <input className="memory-edit-input" value={editingText} onChange={(event) => setEditingText(event.target.value)} /> : <h3>{item.content}</h3>}<p>{item.source === "manual" ? "手动添加" : "从对话中提取并经你确认"} · {formatDate(item.updated_at)}</p></div><div className="memory-actions">{editingId === item.id ? <><button onClick={() => setEditingId(null)}>取消</button><button className="secondary-button" onClick={() => { void updateMemory(item.id, { content: editingText }); setEditingId(null); }}>保存</button></> : <><button onClick={() => void api<MemoryDetail>(`/api/memories/${item.id}`).then(setAudit)}>审计</button><button onClick={() => { setEditingId(item.id); setEditingText(item.content); }}>编辑</button><button onClick={() => void updateMemory(item.id, { status: item.status === "enabled" ? "disabled" : "enabled" })}>{item.status === "enabled" ? "停用" : "启用"}</button><button className="danger-text" onClick={() => void deleteMemory(item.id)}>删除</button></>}</div><span className={`memory-state ${item.status}`}>{item.status === "enabled" ? "使用中" : "已停用"}</span></article>)}</div>}</section>
  </div>;
}

function ToolsView() {
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [runs, setRuns] = useState<ToolRun[]>([]);
  const [schemas, setSchemas] = useState<Record<string, Record<string, unknown>>>({});
  const [url, setUrl] = useState("");
  const [webResult, setWebResult] = useState<{ title: string; url: string; text: string; accessed_at: string } | null>(null);
  const [fetching, setFetching] = useState(false);
  const [error, setError] = useState("");
  async function load() {
    const [definitions, traces] = await Promise.all([
      api<ToolDefinition[]>("/api/tools/catalog"),
      api<ToolRun[]>("/api/tools/runs?limit=20"),
    ]);
    setTools(definitions); setRuns(traces);
  }
  useEffect(() => { void load(); }, []);
  async function loadSchema(name: string) {
    if (schemas[name]) { setSchemas((current) => { const next = { ...current }; delete next[name]; return next; }); return; }
    const result = await api<{ name: string; input_schema: Record<string, unknown> }>(`/api/tools/${encodeURIComponent(name)}/schema`);
    setSchemas((current) => ({ ...current, [name]: result.input_schema }));
  }
  async function fetchPage() {
    if (!url.startsWith("https://")) return;
    setFetching(true); setError(""); setWebResult(null);
    try {
      setWebResult(await api("/api/web/fetch", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) }));
      await load();
    } catch (reason) { setError(reason instanceof Error ? reason.message : "网页读取失败"); }
    finally { setFetching(false); }
  }
  return <div className="page tools-page">
    <PageHead eyebrow="TOOL RUNTIME" title="Tool 中心" description="路由阶段只读取用途、权限和参数摘要；选中 Tool 后才加载完整 Schema，并保留真实运行记录。" />
    <section className="agent-flow"><span>渐进式 Schema</span><div><b>{tools.length} 个 Tool 摘要</b><i>→</i><b>选择候选 Tool</b><i>→</i><b>加载完整参数 Schema</b><i>→</i><b>校验后执行</b></div><p>这样可以减少无关参数进入上下文。这里展示的是可审计路由信息，不是模型私密思维过程。</p></section>
    <section className="web-fetch-box"><div><span>WEB.FETCH · 真实能力</span><h2>读取一个明确的公开网页</h2><p>只允许 HTTPS，并阻止本机、局域网和保留地址。</p></div><input value={url} onChange={(event) => setUrl(event.target.value)} placeholder="https://example.com/article" /><button className="primary-button" disabled={fetching || !url.startsWith("https://")} onClick={() => void fetchPage()}>{fetching ? "读取中…" : "读取网页"}</button></section>
    {error && <div className="lab-error">{error}</div>}
    {webResult && <article className="web-result"><span>网页读取结果</span><h3>{webResult.title}</h3><a href={webResult.url} target="_blank" rel="noreferrer">{webResult.url}</a><p>{webResult.text.slice(0, 1200)}{webResult.text.length > 1200 ? "…" : ""}</p></article>}
    <div className="tools-layout"><section><h2>能力目录 · 当前只加载摘要</h2><div className="tool-grid">{tools.map((tool) => <article key={tool.name} className={tool.availability}><div><code>{tool.name}</code><span>{tool.availability === "available" ? "可用" : "待配置"}</span></div><p>{tool.description}</p><footer><b>{tool.timeout_seconds}s</b>{tool.network_scopes.length > 0 && <b>联网：{tool.network_scopes.join("、")}</b>}{tool.confirmation_required && <b>需要确认</b>}{tool.parameter_summary?.map((item) => <b key={item.name}>{item.name}{item.required ? "*" : ""}</b>)}</footer><button className="schema-toggle" onClick={() => void loadSchema(tool.name)}>{schemas[tool.name] ? "收起完整 Schema" : "按需加载完整 Schema"}</button>{schemas[tool.name] && <pre className="tool-schema">{JSON.stringify(schemas[tool.name], null, 2)}</pre>}{tool.unavailable_reason && <small>{tool.unavailable_reason}</small>}</article>)}</div></section><aside><h2>最近运行</h2>{runs.length ? runs.map((run) => <div className="tool-run" key={run.id}><i className={run.status} /><span><strong>{run.tool_name}</strong><small>{formatDate(run.created_at)} · {run.duration_ms ?? "—"} ms{run.error_code ? ` · ${run.error_code}` : ""}</small></span></div>) : <p className="history-empty">还没有 Tool 运行记录</p>}</aside></div>
    <div className="web-search-note"><b>仍然存在的边界</b><p>当前已实现摘要目录和选中后加载 Schema，并在 Agent Trace 中估算选中 Schema Token；尚未建立离线路由集来量化节省比例和路由准确率。</p></div>
  </div>;
}
function SkillsView() {
  const [skills, setSkills] = useState<SkillItem[]>([]);
  const [selected, setSelected] = useState<SkillItem | null>(null);
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [instructions, setInstructions] = useState("");
  const [tools, setTools] = useState("");
  const [error, setError] = useState("");
  const [routeQuestion, setRouteQuestion] = useState("请上网查一下今天的 Agent 新闻");
  const [routePreview, setRoutePreview] = useState<RoutePreview | null>(null);
  async function load() {
    const items = await api<SkillItem[]>("/api/skills");
    setSkills(items);
    setSelected((current) => current ? items.find((item) => item.id === current.id) || null : items[0] || null);
  }
  useEffect(() => { void load(); }, []);
  async function submit() {
    setError("");
    try {
      const created = await api<SkillItem>("/api/skills", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          description,
          instructions,
          tools: tools.split(/[,，\s]+/).map((item) => item.trim()).filter(Boolean),
        }),
      });
      setName(""); setDescription(""); setInstructions(""); setTools(""); setCreating(false);
      await load();
      setSelected(created);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Skill 创建失败");
    }
  }
  async function previewRoute() {
    if (!routeQuestion.trim()) return;
    setRoutePreview(await api<RoutePreview>(`/api/agent/route?q=${encodeURIComponent(routeQuestion.trim())}`));
  }
  const skillPurpose: Record<string, string> = {
    document_skill: "从本地文档找证据，并生成可以点回原文的回答。",
    excel_skill: "保留工作表和单元格结构，用确定性计算回答表格问题。",
    file_search_skill: "按文件名、类型、标签和知识空间找到本地文件。",
    image_skill: "理解图片与 OCR 文字，并支持用自然语言找图。",
    video_skill: "把视频拆成时间戳转写、关键帧和可检索片段；当前仍是实验能力。",
    web_research_skill: "搜索当前公开网页，核验来源后回答实时或外部问题。",
    recommendation_skill: "针对餐饮、出行和本地生活推荐搜索当前网页，并明确时效与不确定性。",
  };
  const skillSteps: Record<string, string[]> = {
    document_skill: ["确认本轮知识空间", "只读取已确认入库的资料副本", "执行 RAG 检索并筛选最小证据集", "模型根据证据组织回答", "引用不足时明确拒答"],
    excel_skill: ["读取工作簿结构", "定位目标工作表与单元格", "使用确定性计算完成统计", "保留单元格级来源", "写出新文件前等待确认"],
    file_search_skill: ["解析文件查找条件", "在允许范围内搜索元数据", "按空间和类型过滤", "返回文件位置与副本状态"],
    image_skill: ["识别画面和 OCR 文字", "生成待确认标题与标签", "确认后写入图片索引", "按语义、OCR 和标签检索", "返回图片来源"],
    video_skill: ["确认本地视频来源", "读取媒体元数据", "经确认后转写与抽帧", "生成时间戳索引", "回答时标注语音/OCR/画面来源"],
    web_research_skill: ["判断问题是否需要当前公网信息", "调用 web.search 获取候选网页", "保留标题、网址和发布时间", "模型只基于网页摘要组织回答", "引用校验失败时拒绝展示结论"],
    recommendation_skill: ["提取地点、时间和偏好约束", "搜索当前可核验的本地生活来源", "筛选具体地点并保留地址或商圈", "区分事实、评价和建议", "无法核验营业状态或评分时明确说明"],
  };
  const selectedSteps = selected ? (skillSteps[selected.id] || selected.steps || []) : [];
  return <div className="page skills-page">
    <PageHead eyebrow="SKILL SYSTEM" title="Skill 中心" description="Skill 是 Agent 的可复用工作流；SKILL.md 定义触发条件和步骤，skill.json 定义 Tool、权限、超时与错误。" action={<button className="primary-button" onClick={() => setCreating(!creating)}>{creating ? "取消创建" : "＋ 创建 Skill"}</button>} />
    <section className="route-playground"><div><span>路由与 Plan 预览</span><h2>输入一句话，看 Agent 为什么选择这套能力</h2><p>展示的是可审计决策，不是隐藏思维：请求类型、路由理由、Skill、Tool 和预期任务流。</p></div><div className="route-input"><input value={routeQuestion} onChange={(event) => setRouteQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void previewRoute(); }} /><button className="primary-button" onClick={() => void previewRoute()}>生成 Plan</button></div>{routePreview && <><div className="route-result"><b>{routePreview.question}</b><i>→</i><span>意图<strong>{routePreview.intent}</strong></span><i>→</i><span>Skill<strong>{routePreview.skill}</strong></span><i>→</i><span>Tools<strong>{routePreview.tools.join("、") || "无需 Tool"}</strong></span>{routePreview.memory_candidate && <em>{routePreview.memory_candidate.status === "enabled" ? "将写入长期 Memory" : "将生成待确认记忆"}：{routePreview.memory_candidate.content}</em>}</div>{routePreview.plan && <PlanFlow plan={routePreview.plan} />}</>}</section>
    {creating && <section className="skill-builder">
      <div><label>Skill 名称（小写英文和连字符）<input value={name} onChange={(event) => setName(event.target.value.toLowerCase().replace(/_/g, "-"))} placeholder="study-notes" /></label><label>触发描述<textarea value={description} onChange={(event) => setDescription(event.target.value)} placeholder="说明它做什么，以及用户在什么情况下会触发它。" /></label></div>
      <div><label>工作流指令<textarea value={instructions} onChange={(event) => setInstructions(event.target.value)} placeholder={"1. 确认输入范围\n2. 调用必要 Tool\n3. 校验结果并返回来源"} /></label><label>允许调用的 Tools<input value={tools} onChange={(event) => setTools(event.target.value)} placeholder="rag.search, file.search" /></label></div>
      <footer><span>{error || "将生成标准 YAML frontmatter、SKILL.md 和运行契约；写操作默认要求确认。"}</span><button className="primary-button" disabled={!name || description.length < 10 || instructions.length < 10} onClick={() => void submit()}>生成 Skill 文件</button></footer>
    </section>}
    <div className="skills-layout">
      <div className="skill-grid">{skills.map((skill) => <button key={skill.id} className={`skill-card ${selected?.id === skill.id ? "selected" : ""}`} onClick={() => setSelected(skill)}><span>{skill.source === "builtin" ? "内置" : "个人"}</span><h3>{skill.name}</h3><p>{skill.description}</p><div><b>{skill.tools.length} Tools</b><b>{skill.has_skill_md ? "SKILL.md ✓" : "缺少 SKILL.md"}</b></div></button>)}</div>
      {selected && <aside className="skill-inspector"><span>SKILL EXPLAINER</span><h2>{selected.name}</h2><p>{skillPurpose[selected.id] || selected.description}</p><div className="skill-role-flow"><span><b>何时触发</b><small>{selected.description}</small></span><i>→</i><span><b>执行流程</b><small>{selectedSteps.length} 个可审计步骤</small></span><i>→</i><span><b>调用 Tool</b><small>{selected.tools.join("、") || "无需 Tool"}</small></span><i>→</i><span><b>返回结果</b><small>{selected.output_fields.join("、") || "结构化结果"}</small></span></div><section className="skill-step-panel"><header><b>它具体会做什么</b><small>按顺序执行；失败会停在对应步骤</small></header><ol>{selectedSteps.map((step, index) => <li key={`${selected.id}-${index}`}><i>{index + 1}</i><span>{step}</span></li>)}</ol></section><div className="skill-boundary"><b>权限边界</b><dl><div><dt>可读取</dt><dd>{selected.read_scope.join("、") || "未声明"}</dd></div><div><dt>可写入</dt><dd>{selected.write_scope.join("、") || "无"}</dd></div><div><dt>写入确认</dt><dd>{selected.requires_confirmation_for_write ? "必须由你确认" : "不需要"}</dd></div><div><dt>超时</dt><dd>{selected.timeout_seconds ? `${selected.timeout_seconds} 秒` : "未声明"}</dd></div></dl></div>{!!selected.recoverable_errors.length && <div className="skill-errors"><b>失败时会明确报告</b><p>{selected.recoverable_errors.map((error) => <code key={error}>{error}</code>)}</p></div>}<details><summary>高级：查看原始 SKILL.md</summary><pre>{selected.content || "该旧版 Skill 尚未补齐 SKILL.md。"}</pre></details></aside>}
    </div>
  </div>;
}

function RagLabView({ activeSpace, documents }: { activeSpace?: Space; documents: DocumentItem[] }) {
  const [cases, setCases] = useState<EvaluationCase[]>([]);
  const [question, setQuestion] = useState("");
  const [documentId, setDocumentId] = useState("");
  const [locator, setLocator] = useState("");
  const [topK, setTopK] = useState(5);
  const [evaluationLimit, setEvaluationLimit] = useState(2);
  const [result, setResult] = useState<EvaluationRun | null>(null);
  const [agentQuality, setAgentQuality] = useState<AgentQualityResult | null>(null);
  const [agentRunning, setAgentRunning] = useState(false);
  const [running, setRunning] = useState(false);
  const [datasetStatus, setDatasetStatus] = useState("");
  const [error, setError] = useState("");
  async function load() {
    if (!activeSpace) return;
    const helloAgents = documents.find((item) => item.original_name === "Hello-Agents-V1.0.2-20260210.pdf");
    if (helloAgents) {
      const imported = await api<{ total: number; imported: number }>("/api/rag/evaluation/import/hello-agents" + `?space_id=${encodeURIComponent(activeSpace.id)}`, { method: "POST" });
      setDatasetStatus(`Hello-Agents 人工评估集已就绪：${imported.total} 题`);
    } else {
      setDatasetStatus("导入 Hello-Agents PDF 并完成索引后，可自动加载配套 30 题");
    }
    setCases(await api<EvaluationCase[]>(`/api/rag/evaluation/cases?space_id=${encodeURIComponent(activeSpace.id)}`));
  }
  useEffect(() => {
    setDocumentId(documents[0]?.id || "");
    setResult(null);
    void load();
  }, [activeSpace?.id, documents.length]);
  async function addCase() {
    if (!activeSpace || !question.trim() || !documentId) return;
    setError("");
    try {
      await api("/api/rag/evaluation/cases", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, expected_document_id: documentId, expected_locator: locator, space_id: activeSpace.id }),
      });
      setQuestion(""); setLocator("");
      await load();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "添加失败");
    }
  }
  async function deleteCase(id: string) {
    await api(`/api/rag/evaluation/cases/${id}`, { method: "DELETE" });
    await load();
  }
  async function runEvaluation() {
    if (!activeSpace) return;
    setRunning(true); setError("");
    try {
      setResult(await api<EvaluationRun>("/api/rag/evaluation/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ space_id: activeSpace.id, top_k: topK, limit: Math.min(evaluationLimit, cases.length) }),
      }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "评估失败");
    } finally {
      setRunning(false);
    }
  }
  async function runAgentQuality() {
    setAgentRunning(true); setError("");
    try { setAgentQuality(await api<AgentQualityResult>("/api/agent/evaluation/run", { method: "POST" })); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "Agent 质量评估失败"); }
    finally { setAgentRunning(false); }
  }
  const qualityNames: Record<string, string> = {
    routing_accuracy: "路由 / Skill 准确率", refusal_accuracy: "空证据拒答率",
    citation_location_success: "引用定位成功率", agent_task_success: "Agent 任务成功率", claim_support_rate: "Claim—引用支持率",
  };
  const metrics = [
    ["Recall@K", "该找到的正确来源，有多少在前 K 条中被找到了。漏掉正确资料时它会下降。", "检索覆盖率"],
    ["MRR", "第一个正确来源越靠前越好；第 1 名得 1 分，第 2 名得 0.5 分。", "首个答案排名"],
    ["nDCG@K", "同时衡量是否命中和排名位置，正确来源排得越靠前，分数越高。", "排序质量"],
    ["P95 延迟", "95% 的检索请求能在这个时间内完成，比平均值更能暴露偶发卡顿。", "工程稳定性"],
  ];
  return <div className="page rag-lab-page">
    <PageHead eyebrow="RAG EVALUATION LAB" title="RAG 实验室" description="用人工标注问题验证检索，而不是凭一次回答主观判断。指标只评价“找资料”，不等于最终回答完全正确。" action={<div className="run-eval"><select value={topK} onChange={(event) => setTopK(Number(event.target.value))}>{[1, 3, 5, 10].map((value) => <option key={value} value={value}>Top K = {value}</option>)}</select><select value={evaluationLimit} onChange={(event) => setEvaluationLimit(Number(event.target.value))}>{[2, 5, 10, 30].map((value) => <option key={value} value={value}>评估数量 = {value}</option>)}</select><button className="primary-button" disabled={running || !cases.length} onClick={() => void runEvaluation()}>{running ? "评估中…" : `运行 ${Math.min(evaluationLimit, cases.length)} 条评估`}</button></div>} />
    <section className="agent-quality-gate"><header><div><span>AGENT QUALITY GATES</span><h2>95% 不是口号，要逐项过门槛</h2><p>路由与拒答可自动运行；引用定位来自人工 RAG 题；Claim 支持率必须由人判断“来源是否真的支持这句话”。</p></div><button className="primary-button" disabled={agentRunning} onClick={() => void runAgentQuality()}>{agentRunning ? "验证中…" : "运行 Agent 验证"}</button></header>{agentQuality ? <><div className="quality-metrics">{Object.entries(agentQuality.metrics).map(([key, metric]) => { const passed = metric.value !== null && metric.value >= metric.target; return <article className={metric.value === null ? "waiting" : passed ? "passed" : "failed"} key={key}><span>{qualityNames[key] || key}</span><strong>{metric.value === null ? "待标注" : `${(metric.value * 100).toFixed(1)}%`}</strong><small>{metric.case_count} 题 · 目标 ≥{metric.target * 100}%</small><i>{metric.value === null ? "需要人工集" : passed ? "通过" : "未通过"}</i></article>; })}</div><div className={`quality-release ${agentQuality.publish_ready ? "ready" : "blocked"}`}><b>{agentQuality.publish_ready ? "✓ 可以发布 95% 指标" : "尚不能宣称达到 95%"}</b><p>{agentQuality.publish_ready ? agentQuality.notes : agentQuality.publication_blockers.join("；")}</p></div><details className="quality-cases"><summary>查看逐题路由与拒答结果</summary>{[...agentQuality.route_details, ...agentQuality.refusal_details].map((item, index) => <div key={index} className={item.passed ? "passed" : "failed"}><i>{item.passed ? "✓" : "×"}</i><span><b>{item.question}</b><small>{"expected_skill" in item ? `预期 ${item.expected_skill} · 实际 ${item.actual_skill}` : item.behavior}</small></span></div>)}</details></> : <div className="quality-empty">点击运行后，会显示每项样本数、通过率、失败题和发布阻塞项；不会用空数据凑 95%。</div>}</section>
    <div className="metric-explainers">{metrics.map(([name, text, focus]) => <article key={name}><span>{focus}</span><h3>{name}</h3><p>{text}</p></article>)}</div>
    <div className={`evaluation-dataset-status ${cases.length >= 30 ? "ready" : ""}`}><span>{cases.length >= 30 ? "✓" : "i"}</span><div><strong>{datasetStatus || "正在核对评估集…"}</strong><p>题目只参与检索，参考答案不会送给检索器；当前空间共 {cases.length} 条人工标注。</p></div></div>
    <section className="evaluation-builder"><div><h2>人工标注问题</h2><p>先写一个答案确实存在于资料中的问题，再指定正确文档；如果知道页码，可继续填写定位。</p></div><input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="例如：教程把 Agent 工作流分成哪些步骤？" /><select value={documentId} onChange={(event) => setDocumentId(event.target.value)}><option value="">选择正确来源文档</option>{documents.map((document) => <option value={document.id} key={document.id}>{document.title}</option>)}</select><input value={locator} onChange={(event) => setLocator(event.target.value)} placeholder="可选：第 12 页" /><button className="secondary-button" disabled={!question.trim() || !documentId} onClick={() => void addCase()}>＋ 加入评估集</button></section>
    {error && <div className="lab-error">{error}</div>}
    {!!cases.length && <div className="evaluation-cases">{cases.map((item) => <article key={item.id}><div><strong>{item.question}</strong><p>正确来源：{item.expected_document_title}{item.expected_locator ? ` · ${item.expected_locator}` : ""}</p></div><button onClick={() => void deleteCase(item.id)}>删除</button></article>)}</div>}
    {result ? <section className="evaluation-result"><div className="result-metrics"><MetricValue name={`Recall@${result.top_k}`} value={result.recall} /><MetricValue name="MRR" value={result.mrr} /><MetricValue name="nDCG" value={result.ndcg} /><MetricValue name="平均延迟" value={`${result.mean_latency_ms} ms`} /><MetricValue name="P95 延迟" value={`${result.p95_latency_ms} ms`} /></div><h2>逐题结果</h2>{result.details.map((item) => <article key={item.case_id} className={item.hit ? "hit" : "miss"}><span>{item.hit ? "命中" : "未命中"}</span><div><strong>{item.question}</strong><p>{item.hit ? `正确来源排名：第 ${item.rank} 名` : "Top K 中没有找到人工标注来源"} · {item.latency_ms} ms</p></div></article>)}</section>
      : !cases.length && <div className="lab-start"><b>从 3–10 条高质量问题开始</b><p>小数据集适合调试，不足以写成项目准确率。发布指标前建议积累至少 100 条人工标注问题。</p></div>}
  </div>;
}

function MetricValue({ name, value }: { name: string; value: number | string }) {
  return <div><span>{name}</span><strong>{typeof value === "number" ? `${Math.round(value * 100)}%` : value}</strong></div>;
}

function ImageSearchView({ activeSpace, openFiles }: { activeSpace?: Space; openFiles: () => void }) {
  const [query, setQuery] = useState("");
  const [items, setItems] = useState<ImageAsset[]>([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<ImageAsset | null>(null);
  async function load(value = query) {
    if (!activeSpace) return;
    setLoading(true);
    try {
      setItems(await api<ImageAsset[]>(`/api/images?space_id=${encodeURIComponent(activeSpace.id)}&q=${encodeURIComponent(value)}`));
    } finally {
      setLoading(false);
    }
  }
  useEffect(() => {
    void load("");
  }, [activeSpace?.id]);
  return <div className="page image-page">
    <PageHead eyebrow="MULTIMODAL SEARCH" title="图片搜索" description={`在“${activeSpace?.name || "当前空间"}”中按画面含义、图中文字和标签搜索。`} action={<button className="primary-button" onClick={openFiles}>＋ 添加图片</button>} />
    <div className="image-search"><span>⌕</span><input value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") void load(); }} placeholder="例如：包含 Agent 工作流的紫色架构图" /><button onClick={() => void load()}>{loading ? "搜索中…" : "搜索"}</button></div>
    <div className="image-privacy">图片原件保存在本机；首次理解会把压缩副本发送至阿里云百炼视觉模型，生成描述、OCR 和语义索引。</div>
    {!items.length ? <EmptyState title={loading ? "正在读取图片索引" : "当前空间还没有可搜索图片"} text="添加 PNG/JPG 后，坤坤会识别画面和文字；处理完成后会自动出现在这里。" action={<button className="primary-button" onClick={openFiles}>添加图片</button>} /> : <div className="real-image-grid">{items.map((item) => <button className="real-image-card" key={item.document_id} onClick={() => setSelected(item)}><img src={`${API_BASE}/api/files/${item.document_id}`} alt={item.description || item.title} /><div><strong>{item.title}</strong><p>{item.description}</p><span>{item.tags.map((tag) => <i key={tag}>{tag}</i>)}</span>{query && <small>语义匹配 {Math.round(item.score * 100)}%</small>}</div></button>)}</div>}
    {selected && <div className="modal-backdrop"><section className="image-detail-modal"><button className="modal-close" onClick={() => setSelected(null)}>×</button><img src={`${API_BASE}/api/files/${selected.document_id}`} alt={selected.description || selected.title} /><div><span>图片理解</span><h2>{selected.title}</h2><p>{selected.description}</p><h3>识别文字</h3><pre>{selected.ocr_text || "未识别到清晰文字"}</pre><small>{selected.width} × {selected.height} · {selected.original_name}</small></div></section></div>}
  </div>;
}

type ExperimentHistoryProps = {
  experiments: Experiment[];
  leaderboardExperiments: Experiment[];
  compareExperiments: Experiment[];
  compareExperimentIds: string[];
  experimentView: "leaderboard" | "compare";
  onViewChange: (view: "leaderboard" | "compare") => void;
  onToggleCompare: (id: string) => void;
  onOpenExperiment: (id: string) => void;
  indexes: IndexGeneration[];
};

function ExperimentHistory({
  experiments, leaderboardExperiments, compareExperiments, compareExperimentIds, experimentView,
  onViewChange, onToggleCompare, onOpenExperiment, indexes,
}: ExperimentHistoryProps) {
  const maxLatency = Math.max(1, ...compareExperiments.map((item) => Number(item.summary.latency_ms?.p95 || 0)));
  const metrics: Array<{ label: string; max: number; value: (item: Experiment) => number | undefined; format: (value?: number) => string }> = [
    { label: "Document Recall@5", max: 1, value: (item) => item.summary.document_recall?.["5"], format: metricPercent },
    { label: "Evidence Recall@1", max: 1, value: (item) => item.summary.evidence_recall?.["1"], format: metricPercent },
    { label: "Evidence Recall@5", max: 1, value: (item) => item.summary.evidence_recall?.["5"], format: metricPercent },
    { label: "Evidence Recall@10", max: 1, value: (item) => item.summary.evidence_recall?.["10"], format: metricPercent },
    { label: "MRR", max: 1, value: (item) => item.summary.mrr, format: metricNumber },
    { label: "nDCG@10", max: 1, value: (item) => item.summary.ndcg_10, format: metricNumber },
    { label: "Citation resolvable", max: 1, value: (item) => item.summary.citation_resolvable_rate, format: metricPercent },
    { label: "P50 latency", max: maxLatency, value: (item) => item.summary.latency_ms?.p50, format: metricMillis },
    { label: "P95 latency", max: maxLatency, value: (item) => item.summary.latency_ms?.p95, format: metricMillis },
    { label: "P99 latency", max: maxLatency, value: (item) => item.summary.latency_ms?.p99, format: metricMillis },
  ];
  const baseline = compareExperiments[0];
  return <section className="infra-panel experiment-history">
    <header className="experiment-history-header">
      <div><span>RUN HISTORY / LEADERBOARD</span><h2>实验运行榜单与对比矩阵</h2><p className="experiment-history-caption">每次运行都保存为独立快照；质量、延迟、配置和 Bad Case 可回看，不会覆盖之前的结果。</p></div>
      <div className="experiment-history-actions">
        <button className={experimentView === "leaderboard" ? "selected" : ""} onClick={() => onViewChange("leaderboard")}>排行榜</button>
        <button className={experimentView === "compare" ? "selected" : ""} onClick={() => onViewChange("compare")}>对比矩阵 {compareExperiments.length ? `(${compareExperiments.length})` : ""}</button>
        <button className="primary-button" disabled={compareExperiments.length < 2} onClick={() => onViewChange("compare")}>可视化对比</button>
      </div>
    </header>
    <div className="experiment-history-summary">
      <div><span>已保存运行</span><strong>{experiments.length}</strong><small>包含成功、失败和排队记录</small></div>
      <div><span>可用于排行</span><strong>{leaderboardExperiments.length}</strong><small>已完成且有 Gold 指标</small></div>
      <div><span>当前选择</span><strong>{compareExperiments.length} / 3</strong><small>选择后点击“可视化对比”</small></div>
      <div><span>排序口径</span><strong>0.5 / 0.3 / 0.2</strong><small>Evidence R@5 · MRR · nDCG@10</small></div>
    </div>
    {experimentView === "leaderboard" ? <div className="table-responsive experiment-history-table-wrap"><div className="experiment-history-table">
      <div className="experiment-history-head"><span>选择</span><span>排名</span><span>运行 / 配置</span><span>Pipeline</span><span>Evidence R@5</span><span>MRR</span><span>nDCG@10</span><span>P95</span><span>状态</span><span>操作</span></div>
      {leaderboardExperiments.map((item, index) => {
        const generation = indexes.find((entry) => entry.id === String(item.config?.generation_id || ""));
        const model = generation?.model || String(item.config?.embedding_model || (item.config?.pipeline === "bm25" ? "本地 BM25" : "向量索引"));
        return <div className={`experiment-history-row ${compareExperimentIds.includes(item.id) ? "selected" : ""}`} key={item.id}>
          <label className="experiment-check"><input type="checkbox" checked={compareExperimentIds.includes(item.id)} onChange={() => onToggleCompare(item.id)} /><span className="sr-only">选择 {item.name}</span></label>
          <strong className="experiment-rank">#{index + 1}</strong>
          <span className="experiment-run-name"><b>{item.name}</b><small>{formatDate(item.created_at)} · {model}{generation ? ` · ${generation.dimension}d` : ""}</small></span>
          <span>{experimentPipelineLabel(item)}</span>
          <strong>{metricPercent(item.summary.evidence_recall?.["5"])}</strong>
          <strong>{metricNumber(item.summary.mrr)}</strong>
          <strong>{metricNumber(item.summary.ndcg_10)}</strong>
          <span>{metricMillis(item.summary.latency_ms?.p95)}</span>
          <span className={`infra-status ${item.status}`}>{item.status}</span>
          <button className="link-button" onClick={() => onOpenExperiment(item.id)}>查看</button>
        </div>;
      })}
      {!leaderboardExperiments.length && <p className="infra-empty">还没有成功的 Gold 评测运行。完成一次实验后，这里会自动形成可复现榜单。</p>}
    </div></div> : compareExperiments.length < 2 ? <div className="experiment-compare-empty"><strong>先在排行榜勾选至少 2 次运行</strong><p>建议选择同一数据集、只改变一个变量的运行，例如 1024d 与 256d，或 Hybrid 与 Hybrid + Rerank。</p><button onClick={() => onViewChange("leaderboard")}>返回排行榜选择</button></div> : <div className="experiment-comparison">
      <div className="experiment-compare-context"><span>对比基线</span><strong>{baseline?.name}</strong><small>左侧第一项作为基线，其余运行显示相对差值。</small></div>
      <div className="experiment-compare-cards">{compareExperiments.map((item, index) => <article key={item.id} className={index === 0 ? "baseline" : ""}><span>{index === 0 ? "BASELINE" : `CANDIDATE ${index}`}</span><strong>{item.name}</strong><small>{experimentPipelineLabel(item)} · {String(item.config?.split || "all")} · {item.summary.case_count || 0} cases</small><small>config {String(item.config_hash || "—").slice(0, 10)} · git {String(item.git_revision || "—").slice(0, 10)}</small><a href={`${API_BASE}/api/eval/experiments/${item.id}/report?format=markdown`} target="_blank" rel="noreferrer">打开完整报告 ↗</a></article>)}</div>
      <div className="experiment-metric-matrix" role="table" aria-label="实验指标对比矩阵"><div className="experiment-matrix-head"><strong>指标</strong>{compareExperiments.map((item) => <span key={item.id}>{item.name}</span>)}</div>{metrics.map((metric) => <div className="experiment-matrix-row" key={metric.label}><strong>{metric.label}</strong>{compareExperiments.map((item) => { const value = metric.value(item); const numeric = typeof value === "number" ? value : 0; const width = Math.max(0, Math.min(100, numeric / metric.max * 100)); return <span key={item.id}><b>{metric.format(value)}</b><i><em style={{ width: `${width}%` }} /></i></span>; })}</div>)}</div>
      <div className="experiment-delta-list">{baseline && compareExperiments.slice(1).map((item) => { const qualityDelta = (item.summary.evidence_recall?.["5"] || 0) - (baseline.summary.evidence_recall?.["5"] || 0); const latencyDelta = (item.summary.latency_ms?.p95 || 0) - (baseline.summary.latency_ms?.p95 || 0); return <article key={item.id}><span><b>{item.name}</b><small>相对 {baseline.name}</small></span><strong className={qualityDelta >= 0 ? "positive" : "negative"}>Evidence R@5 {qualityDelta >= 0 ? "+" : ""}{(qualityDelta * 100).toFixed(1)} pp</strong><strong className={latencyDelta <= 0 ? "positive" : "negative"}>P95 {latencyDelta >= 0 ? "+" : ""}{Math.round(latencyDelta)} ms</strong></article>; })}</div>
    </div>}
  </section>;
}

function InfraView({ activeSpace, health }: { activeSpace?: Space; health: Health | null }) {
  const [tab, setTab] = useState<"cockpit" | "traces" | "experiments" | "duel" | "gate">("cockpit");
  const [overview, setOverview] = useState<InfraOverview | null>(null);
  const [jobs, setJobs] = useState<InfraJob[]>([]);
  const [traces, setTraces] = useState<InfraTrace[]>([]);
  const [indexes, setIndexes] = useState<IndexGeneration[]>([]);
  const [datasets, setDatasets] = useState<EvalDataset[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [benchmarks, setBenchmarks] = useState<PerformanceBenchmark[]>([]);
  const [cloudPolicies, setCloudPolicies] = useState<DocumentCloudPolicy[]>([]);
  const [infraBudget, setInfraBudget] = useState<InfraBudget>({ max_api_requests_per_run: 500, max_embedding_input_characters: 5_000_000, allow_multi_model_rebuild: true });
  const [selectedTrace, setSelectedTrace] = useState<InfraTrace | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<EvalDataset | null>(null);
  const [selectedExperiment, setSelectedExperiment] = useState<Experiment | null>(null);
  const [experimentView, setExperimentView] = useState<"leaderboard" | "compare">("leaderboard");
  const [compareExperimentIds, setCompareExperimentIds] = useState<string[]>([]);
  const [plannedIndex, setPlannedIndex] = useState<IndexGeneration | null>(null);
  const [duel, setDuel] = useState<DuelResult | null>(null);
  const [regression, setRegression] = useState<RegressionResult | null>(null);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  const [indexModel, setIndexModel] = useState("qwen3.7-text-embedding");
  const [indexDimension, setIndexDimension] = useState(1024);
  const [indexStrategy, setIndexStrategy] = useState("flat");
  const [chunkPreset, setChunkPreset] = useState("700:120");
  const [experimentPipeline, setExperimentPipeline] = useState("bm25");
  const [experimentIndex, setExperimentIndex] = useState("");
  const [duelQuestion, setDuelQuestion] = useState("Agent 的记忆机制如何区分短期记忆与长期记忆？");
  const [duelLeft, setDuelLeft] = useState("bm25");
  const [duelRight, setDuelRight] = useState("hybrid");
  const [baselineId, setBaselineId] = useState("");
  const [candidateId, setCandidateId] = useState("");

  const spaceId = activeSpace?.id || "ai-agent-learning";
  const readyIndexes = indexes.filter((item) => item.status === "ready");
  const successfulExperiments = experiments.filter((item) => item.status === "succeeded");
  const leaderboardExperiments = useMemo(
    () => [...successfulExperiments].sort((left, right) => experimentQualityScore(right) - experimentQualityScore(left)),
    [successfulExperiments],
  );
  const compareExperiments = compareExperimentIds.map((id) => experiments.find((item) => item.id === id)).filter(Boolean) as Experiment[];
  const selectedIndexId = experimentIndex || readyIndexes[0]?.id || "";

  async function loadInfra() {
    try {
      const [overviewData, jobData, traceData, indexData, datasetData, experimentData, benchmarkData, policyData, budgetData] = await Promise.all([
        api<InfraOverview>("/api/infra/overview"),
        api<InfraJob[]>("/api/infra/jobs?limit=40"),
        api<InfraTrace[]>("/api/infra/traces?limit=50"),
        api<IndexGeneration[]>(`/api/infra/index-generations?space_id=${encodeURIComponent(spaceId)}`),
        api<EvalDataset[]>(`/api/eval/datasets?space_id=${encodeURIComponent(spaceId)}`),
        api<Experiment[]>("/api/eval/experiments?limit=100"),
        api<PerformanceBenchmark[]>("/api/infra/performance-benchmarks?limit=20"),
        api<DocumentCloudPolicy[]>(`/api/settings/privacy/documents?space_id=${encodeURIComponent(spaceId)}`),
        api<InfraBudget>("/api/infra/budget"),
      ]);
      setOverview(overviewData); setJobs(jobData); setTraces(traceData); setIndexes(indexData);
      setDatasets(datasetData); setExperiments(experimentData); setBenchmarks(benchmarkData); setCloudPolicies(policyData); setInfraBudget(budgetData);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "AI Infra 数据读取失败");
    }
  }

  useEffect(() => {
    void loadInfra();
    const timer = window.setInterval(() => void loadInfra(), 3000);
    return () => window.clearInterval(timer);
  }, [spaceId]);

  async function runAction(name: string, action: () => Promise<void>) {
    setBusy(name); setNotice("");
    try { await action(); await loadInfra(); }
    catch (error) { setNotice(error instanceof Error ? error.message : "操作失败"); }
    finally { setBusy(""); }
  }

  async function planGeneration() {
    await runAction("plan-index", async () => {
      const [chunkSize, chunkOverlap] = chunkPreset.split(":").map(Number);
      const item = await api<IndexGeneration>("/api/infra/index-generations", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ space_id: spaceId, model: indexModel, dimension: indexDimension, strategy: indexStrategy, chunk_size: chunkSize, chunk_overlap: chunkOverlap }),
      });
      setPlannedIndex(item);
    });
  }

  async function updateCloudPolicy(item: DocumentCloudPolicy, capability: "embedding" | "llm", allowed: boolean) {
    await runAction(`policy-${item.document_id}-${capability}`, async () => {
      await api(`/api/settings/privacy/documents/${item.document_id}`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          embedding_allowed: capability === "embedding" ? allowed : Boolean(item.embedding_allowed),
          llm_allowed: capability === "llm" ? allowed : Boolean(item.llm_allowed),
        }),
      });
    });
  }

  async function saveInfraBudget() {
    await runAction("save-budget", async () => {
      setInfraBudget(await api<InfraBudget>("/api/infra/budget", {
        method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(infraBudget),
      }));
      setNotice("单次实验预算已保存，超限任务会在入队前被阻止。");
    });
  }

  async function buildGeneration() {
    if (!plannedIndex) return;
    const confirmation = window.prompt(`本次预计处理 ${plannedIndex.estimate?.chunk_count || 0} 个 Chunk，${plannedIndex.estimate?.estimated_batches || 0} 个请求批次，缓存命中 ${Math.round((plannedIndex.estimate?.cache_hit_rate || 0) * 100)}%。\n\n请输入：重建此索引`);
    if (confirmation !== "重建此索引") return;
    await runAction("build-index", async () => {
      await api(`/api/infra/index-generations/${plannedIndex.id}/build`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirmation }) });
      setNotice("索引构建任务已入队，可在 Cockpit 查看进度。");
    });
  }

  async function openTrace(id: string) {
    await runAction(`trace-${id}`, async () => setSelectedTrace(await api<InfraTrace>(`/api/infra/traces/${id}`)));
  }

  async function openDataset(id: string) {
    await runAction(`dataset-${id}`, async () => setSelectedDataset(await api<EvalDataset>(`/api/eval/datasets/${id}`)));
  }

  async function importLegacyDataset() {
    await runAction("import-dataset", async () => {
      const item = await api<EvalDataset>("/api/eval/datasets/import-legacy", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ space_id: spaceId, name: "KUN Gold Set", version: "v1" }) });
      setSelectedDataset(item); setNotice(`已导入 ${item.case_count} 条人工题。`);
    });
  }

  async function createCandidateDataset() {
    const name = window.prompt("评测集名称", "KUN Gold Set 100")?.trim();
    if (!name) return;
    await runAction("create-dataset", async () => {
      const item = await api<EvalDataset>("/api/eval/datasets", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ space_id: spaceId, name, version: "v1" }) });
      setSelectedDataset(item);
    });
  }

  async function generateCandidates() {
    if (!selectedDataset) return;
    const confirmation = window.prompt("候选题会把有限文档片段发送给 DeepSeek，生成后仍是 draft。\n\n请输入：生成候选题");
    if (confirmation !== "生成候选题") return;
    await runAction("generate-candidates", async () => {
      await api(`/api/eval/datasets/${selectedDataset.id}/generate-candidates`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ count: 10, confirmation }) });
      setNotice("10 条候选题已进入后台队列，完成后请逐条确认。");
    });
  }

  async function reviewCase(caseId: string, status: "accepted" | "rejected") {
    if (!selectedDataset) return;
    await runAction(`case-${caseId}`, async () => setSelectedDataset(await api<EvalDataset>(`/api/eval/datasets/${selectedDataset.id}/cases/${caseId}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status }),
    })));
  }

  async function startExperiment() {
    const dataset = selectedDataset || datasets.find((item) => (item.accepted_count || 0) > 0);
    if (!dataset) { setNotice("请先选择包含人工确认题的评测集。"); return; }
    if (experimentPipeline !== "bm25" && !selectedIndexId) { setNotice("Dense/Hybrid 实验需要一个已就绪的索引代次。"); return; }
    await runAction("experiment", async () => {
      const name = `${experimentPipeline.toUpperCase()} · ${new Date().toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}`;
      await api("/api/eval/experiments", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        dataset_version_id: dataset.id, name, pipeline: experimentPipeline,
        generation_id: selectedIndexId || null, candidate_k: 20, top_k: 10, rrf_k: 60,
        reranker_model: experimentPipeline === "hybrid_rerank" ? "qwen3-rerank" : null, split: "all",
      }) });
      setNotice("质量实验已入队。运行期间每道题都会保留排名、阶段耗时和 Trace。");
    });
  }

  async function openExperiment(id: string) {
    await runAction(`experiment-${id}`, async () => setSelectedExperiment(await api<Experiment>(`/api/eval/experiments/${id}`)));
  }

  function toggleExperimentForCompare(id: string) {
    setCompareExperimentIds((current) => {
      if (current.includes(id)) return current.filter((item) => item !== id);
      if (current.length >= 3) {
        setNotice("对比视图最多选择 3 次成功运行。");
        return current;
      }
      return [...current, id];
    });
  }

  async function runDuel() {
    if (!duelQuestion.trim()) return;
    await runAction("duel", async () => setDuel(await api<DuelResult>("/api/infra/retrieval-duel", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        question: duelQuestion, space_id: spaceId,
        left: { pipeline: duelLeft, generation_id: duelLeft === "bm25" ? selectedIndexId || null : selectedIndexId, candidate_k: 20, top_k: 10 },
        right: { pipeline: duelRight, generation_id: selectedIndexId, candidate_k: 20, top_k: 10, reranker_model: duelRight === "hybrid_rerank" ? "qwen3-rerank" : null },
      }),
    })));
  }

  async function compareRuns() {
    if (!baselineId || !candidateId) return;
    await runAction("compare", async () => setRegression(await api<RegressionResult>("/api/eval/compare", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ baseline_id: baselineId, candidate_id: candidateId }),
    })));
  }

  async function runBenchmark() {
    await runAction("benchmark", async () => {
      await api("/api/infra/performance-benchmarks", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sizes: [1000, 10000], dimension: 256, query_count: 100, seed: 20260813 }) });
      setNotice("1k / 10k 确定性向量压测已入队；结果不会被当作检索质量结论。");
    });
  }

  const provider = (capability: string) => overview?.providers.find((item) => item.capability === capability);
  const quality = selectedExperiment?.summary || successfulExperiments[0]?.summary;
  const maxSpan = Math.max(1, ...(selectedTrace?.spans || []).map((item) => Number(item.duration_ms || 0)));

  return <div className="page infra-page">
    <PageHead eyebrow="AI INFRA OBSERVABILITY" title="从请求到回归结论，都能被解释" description={`当前空间：${activeSpace?.name || "默认空间"}。质量 Gold 与确定性向量压测严格分开。`} action={<span className={`infra-live ${health ? "online" : ""}`}><i />{health ? "LIVE" : "OFFLINE"}</span>} />
    <nav className="infra-tabs">{[
      ["cockpit", "Cockpit"], ["traces", "Trace Explorer"], ["experiments", "Experiment Studio"], ["duel", "Retrieval Duel"], ["gate", "Regression Gate"],
    ].map(([id, label]) => <button key={id} className={tab === id ? "selected" : ""} onClick={() => setTab(id as typeof tab)}>{label}</button>)}</nav>
    {notice && <div className="infra-notice"><span>{busy ? "运行中" : "状态"}</span>{notice}<button onClick={() => setNotice("")}>×</button></div>}

    {tab === "cockpit" && <>
      <section className="infra-kpis">
        <article><span>Retrieval P95 · 近 50 次</span><strong>{overview?.traces.p95_ms == null ? "—" : `${overview.traces.p95_ms} ms`}</strong><small>真实 Trace，不含模拟值</small></article>
        <article><span>任务队列</span><strong>{Number(overview?.jobs.running || 0)} <em>/ {Number(overview?.jobs.queued || 0)}</em></strong><small>运行中 / 等待中 · 可恢复</small></article>
        <article><span>活跃索引</span><strong>{indexes.filter((item) => item.is_active).length}</strong><small>{readyIndexes.length} 个代次已验证</small></article>
        <article><span>Evidence Recall@5</span><strong>{quality?.evidence_recall ? `${Math.round((quality.evidence_recall["5"] || 0) * 100)}%` : "待评测"}</strong><small>{quality?.case_count ? `${quality.case_count} 条人工 Gold` : "至少 1 条 accepted case"}</small></article>
      </section>
      <section className="infra-provider-strip">
        {[['embedding', 'Embedding', '百炼'], ['rerank', 'Reranker', '百炼'], ['chat', 'Answer LLM', 'DeepSeek']].map(([capability, label, vendor]) => { const item = provider(capability); return <div key={capability}><i className={item?.connection_status === "connected" ? "ok" : item?.configured ? "waiting" : "off"} /><span><b>{label}</b><small>{vendor} · {item?.model || "待配置"}</small></span><strong>{item?.connection_status === "connected" ? "已验证" : item?.configured ? "待验证" : "待配置"}</strong></div>; })}
      </section>
      <section className="infra-pipeline"><header><div><span>REQUEST PIPELINE</span><h2>一次检索请求经过了什么</h2></div><p>每个节点生成独立 Span；失败会保留真实错误码并停止伪装执行。</p></header><div>{[["01","BM25","本地 FTS"],["02","Dense","FAISS"],["03","RRF","融合"],["04","Rerank","百炼"],["05","LLM","DeepSeek"]].map(([n, title, text], index) => <span key={title}><i>{n}</i><b>{title}</b><small>{text}</small>{index < 4 && <em>→</em>}</span>)}</div></section>
      <div className="infra-two-columns">
        <section className="infra-panel"><header><div><span>PERSISTENT RUNNER</span><h2>任务与恢复</h2></div><b>{jobs.filter((item) => ["queued","running","retry_wait"].includes(item.status)).length} ACTIVE</b></header><div className="infra-job-list">{jobs.slice(0, 7).map((job) => <article key={job.id}><i className={`job-${job.status}`} /><div><strong>{job.job_type.replaceAll("_", " ")}</strong><p>{job.message || job.phase}</p><span><b style={{ width: `${job.progress}%` }} /></span></div><aside><b>{job.progress}%</b><small>尝试 {job.attempt}/{job.max_attempts}</small></aside></article>)}{!jobs.length && <p className="infra-empty">队列为空，启动索引或实验后会显示真实进度。</p>}</div></section>
        <section className="infra-panel"><header><div><span>INDEX GENERATIONS</span><h2>索引代次与回滚</h2></div><b>{readyIndexes.length} READY</b></header><div className="infra-index-list">{indexes.slice(0, 6).map((item) => <article key={item.id}><div><strong>{item.model}</strong><p>{item.dimension}d · {item.strategy.toUpperCase()} · Chunk {item.chunk_size}/{item.chunk_overlap}</p></div><span className={`infra-status ${item.status}`}>{item.is_active ? "ACTIVE" : item.status.toUpperCase()}</span><small>{item.vector_count.toLocaleString()} vectors · {formatSize(item.index_bytes || 0)}</small>{item.status === "ready" && !item.is_active && <button onClick={() => void runAction(`activate-${item.id}`, async () => { await api(`/api/infra/index-generations/${item.id}/activate`, { method: "POST" }); })}>切换</button>}</article>)}{!indexes.length && <p className="infra-empty">还没有索引代次。下方配置只会在你确认后产生云端请求。</p>}</div></section>
      </div>
      <section className="infra-builder"><header><div><span>IMMUTABLE INDEX BUILDER</span><h2>建立可验证、可切换的索引代次</h2></div><p>模型、维度、Chunk 与源文档指纹共同决定代次；不会覆盖现有索引。</p></header><div className="infra-form-grid"><label>Embedding<select value={indexModel} onChange={(event) => setIndexModel(event.target.value)}><option value="qwen3.7-text-embedding">qwen3.7-text-embedding</option><option value="text-embedding-v4">text-embedding-v4</option></select></label><label>维度<select value={indexDimension} onChange={(event) => setIndexDimension(Number(event.target.value))}><option value={1024}>1024</option><option value={256}>256</option></select></label><label>Chunk / overlap<select value={chunkPreset} onChange={(event) => setChunkPreset(event.target.value)}><option value="400:80">400 / 80</option><option value="700:120">700 / 120</option><option value="1000:150">1000 / 150</option></select></label><label>FAISS<select value={indexStrategy} onChange={(event) => setIndexStrategy(event.target.value)}><option value="flat">Flat · 精确</option><option value="hnsw">HNSW · ANN</option></select></label><button className="secondary-button" disabled={Boolean(busy)} onClick={() => void planGeneration()}>{busy === "plan-index" ? "计算中…" : "计算重建计划"}</button></div>{plannedIndex && <div className="infra-estimate"><div><span>Chunk</span><strong>{plannedIndex.estimate?.chunk_count || 0}</strong></div><div><span>API 批次</span><strong>{plannedIndex.estimate?.estimated_batches || 0}</strong></div><div><span>缓存命中</span><strong>{Math.round((plannedIndex.estimate?.cache_hit_rate || 0) * 100)}%</strong></div><div><span>重发字符</span><strong>{(plannedIndex.estimate?.estimated_input_characters || 0).toLocaleString()}</strong></div><div><span>费用口径</span><strong>{plannedIndex.estimate?.cost_status === "estimated" ? "估算" : "实际"}</strong></div><button className="primary-button" disabled={Boolean(busy)} onClick={() => void buildGeneration()}>确认并入队</button></div>}</section>
      <section className="infra-panel cloud-policy-panel"><header><div><span>DATA EGRESS POLICY</span><h2>逐份资料授权云端处理</h2></div><b>{cloudPolicies.filter((item) => item.embedding_allowed || item.llm_allowed).length} / {cloudPolicies.length} ALLOWED</b></header><p>默认全部关闭。Embedding 授权控制发送 Chunk 到百炼；LLM 授权控制发送检索片段给 DeepSeek，包括候选题生成。</p><div className="infra-budget-row"><label>单次 API 请求上限<input type="number" min={1} max={10000} value={infraBudget.max_api_requests_per_run} onChange={(event) => setInfraBudget({ ...infraBudget, max_api_requests_per_run: Number(event.target.value) })} /></label><label>单次发送字符上限<input type="number" min={1000} max={100000000} value={infraBudget.max_embedding_input_characters} onChange={(event) => setInfraBudget({ ...infraBudget, max_embedding_input_characters: Number(event.target.value) })} /></label><label><input type="checkbox" checked={infraBudget.allow_multi_model_rebuild} onChange={(event) => setInfraBudget({ ...infraBudget, allow_multi_model_rebuild: event.target.checked })} /> 允许多模型重复建索引</label><button onClick={() => void saveInfraBudget()}>保存预算</button></div><div>{cloudPolicies.map((item) => <article key={item.document_id}><span><strong>{item.title}</strong><small>{item.original_name}</small></span><label><input type="checkbox" checked={Boolean(item.embedding_allowed)} onChange={(event) => void updateCloudPolicy(item, "embedding", event.target.checked)} /> 百炼 Embedding</label><label><input type="checkbox" checked={Boolean(item.llm_allowed)} onChange={(event) => void updateCloudPolicy(item, "llm", event.target.checked)} /> DeepSeek LLM</label></article>)}{!cloudPolicies.length && <p className="infra-empty">当前空间没有资料。添加资料后，必须在这里明确允许才会发送到云端。</p>}</div></section>
    </>}

    {tab === "traces" && <div className="infra-trace-layout"><section className="infra-panel trace-master"><header><div><span>TRACE EXPLORER</span><h2>最近请求</h2></div><b>{traces.length} TRACES</b></header>{traces.map((trace) => <button key={trace.id} className={selectedTrace?.id === trace.id ? "selected" : ""} onClick={() => void openTrace(trace.id)}><i className={trace.status} /><span><strong>{trace.name.replaceAll("_", " ")}</strong><small>{trace.trace_type} · {formatDate(trace.started_at)}</small></span><b>{trace.duration_ms == null ? "running" : `${trace.duration_ms} ms`}</b></button>)}</section><section className="infra-panel trace-detail"><header><div><span>SPAN WATERFALL</span><h2>{selectedTrace?.name.replaceAll("_", " ") || "选择一个 Trace"}</h2></div>{selectedTrace && <b>{selectedTrace.status.toUpperCase()}</b>}</header>{selectedTrace ? <><div className="trace-summary-line"><span>ID {selectedTrace.id.slice(0, 12)}</span><span>{selectedTrace.spans?.length || 0} spans</span><span>{selectedTrace.duration_ms || 0} ms</span></div><div className="span-waterfall">{(selectedTrace.spans || []).map((span) => <article key={span.id}><span><strong>{span.operation}</strong><small>{span.kind}</small></span><div><i className={span.status} style={{ width: `${Math.max(3, Number(span.duration_ms || 0) / maxSpan * 100)}%` }} /></div><b>{span.duration_ms || 0} ms</b></article>)}</div><details><summary>Trace attributes</summary><pre>{JSON.stringify(selectedTrace.attributes, null, 2)}</pre></details></> : <p className="infra-empty">从左侧选择请求，查看 BM25、FAISS、RRF、Rerank、LLM 与缓存阶段的真实耗时。</p>}</section></div>}

    {tab === "experiments" && <>
      <div className="quality-separation"><div><b>QUALITY TRACK</b><strong>人工 Gold · 真实文档</strong><p>用于 Recall、MRR、nDCG、Bad Case；候选题未经确认不计分。</p></div><i>≠</i><div><b>PERFORMANCE TRACK</b><strong>确定性生成向量</strong><p>只用于延迟、QPS、内存和 ANN Recall，不宣称检索准确率。</p></div></div>
      <div className="infra-two-columns experiment-top"><section className="infra-panel"><header><div><span>DATASET VERSIONING</span><h2>Gold 数据集</h2></div><div><button onClick={() => void importLegacyDataset()}>导入现有题</button><button onClick={() => void createCandidateDataset()}>新建</button></div></header><div className="dataset-list">{datasets.map((item) => <button key={item.id} className={selectedDataset?.id === item.id ? "selected" : ""} onClick={() => void openDataset(item.id)}><span><strong>{item.name} <i>{item.version}</i></strong><small>{item.accepted_count || 0} accepted · {item.draft_count || 0} draft</small></span><b>{item.status}</b></button>)}{!datasets.length && <p className="infra-empty">可先导入已有 30 条题，或创建 100 条 Gold 工作集。</p>}</div></section><section className="infra-panel"><header><div><span>EXPERIMENT CONFIG</span><h2>一次只改变一个变量</h2></div><b>REPRODUCIBLE</b></header><div className="experiment-controls"><label>Pipeline<select value={experimentPipeline} onChange={(event) => setExperimentPipeline(event.target.value)}><option value="bm25">BM25</option><option value="dense">Dense</option><option value="hybrid">BM25 + Dense + RRF</option><option value="hybrid_rerank">Hybrid + qwen3-rerank</option></select></label><label>索引代次<select value={selectedIndexId} onChange={(event) => setExperimentIndex(event.target.value)}><option value="">BM25 不使用向量索引</option>{readyIndexes.map((item) => <option value={item.id} key={item.id}>{item.model} · {item.dimension}d · {item.chunk_size}/{item.chunk_overlap}</option>)}</select></label><button className="primary-button" disabled={Boolean(busy) || !datasets.length} onClick={() => void startExperiment()}>{busy === "experiment" ? "入队中…" : "运行质量实验"}</button></div><p className="experiment-rule">每次保存 Dataset hash、配置 hash、Git revision、机器信息和逐题排名。Recall@10 强制检索至少 10 条结果。</p></section></div>
      {selectedDataset && <section className="infra-panel gold-workbench"><header><div><span>HUMAN REVIEW</span><h2>{selectedDataset.name} · 人工 Gold 工作台</h2></div><div><b>{selectedDataset.cases?.filter((item) => item.status === "accepted").length || 0} / 100</b><button disabled={Boolean(busy)} onClick={() => void generateCandidates()}>＋ 生成 10 条 draft</button></div></header><div>{(selectedDataset.cases || []).slice(0, 20).map((item, index) => <article key={item.id}><i>{String(index + 1).padStart(2, "0")}</i><span><strong>{item.question}</strong><small>{item.query_type} · {item.difficulty} · {item.gold?.[0]?.title} {item.gold?.[0]?.locator}</small></span><b className={`case-${item.status}`}>{item.status}</b>{item.status === "draft" && <aside><button onClick={() => void reviewCase(item.id, "rejected")}>拒绝</button><button onClick={() => void reviewCase(item.id, "accepted")}>确认 Gold</button></aside>}</article>)}</div></section>}
      <ExperimentHistory experiments={experiments} leaderboardExperiments={leaderboardExperiments} compareExperiments={compareExperiments} compareExperimentIds={compareExperimentIds} experimentView={experimentView} onViewChange={setExperimentView} onToggleCompare={toggleExperimentForCompare} onOpenExperiment={(id) => void openExperiment(id)} indexes={indexes} />
      {selectedExperiment && <section className="infra-panel bad-cases"><header><div><span>BAD CASE ANALYSIS</span><h2>{selectedExperiment.name}</h2></div><button onClick={() => setSelectedExperiment(null)}>关闭</button></header><div className="metric-row">{[["Document R@5", selectedExperiment.summary.document_recall?.["5"]], ["Evidence R@5", selectedExperiment.summary.evidence_recall?.["5"]], ["MRR", selectedExperiment.summary.mrr], ["nDCG@10", selectedExperiment.summary.ndcg_10], ["Citation", selectedExperiment.summary.citation_resolvable_rate]].map(([label, value]) => <div key={String(label)}><span>{label}</span><strong>{typeof value === "number" ? `${Math.round(value * 1000) / 10}%` : "—"}</strong></div>)}</div><div className="bad-case-list">{(selectedExperiment.cases || []).filter((item) => item.failure_category).map((item) => <article key={item.case_id}><b>{item.failure_category}</b><span><strong>{item.question}</strong><small>{item.latency_ms} ms · 返回 {item.rankings.returned?.length || 0} 个候选</small></span></article>)}{!(selectedExperiment.cases || []).some((item) => item.failure_category) && <p className="infra-empty">本次运行没有 Bad Case。</p>}</div></section>}
    </>}

    {tab === "duel" && <><section className="duel-controls"><div><span>RETRIEVAL DUEL</span><h2>同一道题，逐阶段看排名为什么改变</h2></div><input value={duelQuestion} onChange={(event) => setDuelQuestion(event.target.value)} /><label>A<select value={duelLeft} onChange={(event) => setDuelLeft(event.target.value)}><option value="bm25">BM25</option><option value="dense">Dense</option><option value="hybrid">Hybrid</option></select></label><label>B<select value={duelRight} onChange={(event) => setDuelRight(event.target.value)}><option value="dense">Dense</option><option value="hybrid">Hybrid</option><option value="hybrid_rerank">Hybrid + Rerank</option></select></label><button className="primary-button" disabled={Boolean(busy) || !readyIndexes.length} onClick={() => void runDuel()}>{busy === "duel" ? "对比中…" : "开始对决"}</button></section>{duel ? <div className="duel-grid">{[["A", duel.left, duelLeft], ["B", duel.right, duelRight]].map(([side, result, pipeline]) => { const value = result as DuelResult["left"]; return <section className="infra-panel" key={String(side)}><header><div><span>CONFIG {side as string}</span><h2>{String(pipeline).toUpperCase()}</h2></div><b>{value.duration_ms} ms</b></header><div className="duel-stages">{value.stages.map((stage) => <span key={stage.stage}><b>{stage.stage}</b><i>{stage.duration_ms} ms</i><small>{stage.count} candidates</small></span>)}</div><div className="duel-ranking">{value.results.slice(0, 8).map((item, index) => <article key={item.id}><i>{index + 1}</i><span><strong>{item.title}</strong><p>{String(item.text || "").slice(0, 90)}</p><small>{item.locator} · lexical #{item.lexical_rank || "—"} · dense #{item.vector_rank || "—"} · rerank #{item.rerank_rank || "—"}</small></span><b>{Number(item.score || item.rerank_score || 0).toFixed(4)}</b></article>)}</div></section>; })}</div> : <div className="duel-empty"><strong>选择一个已就绪索引后开始</strong><p>两边分别生成真实 Trace；结果保留 BM25、Dense、Fusion 和 Rerank 的阶段排名。</p></div>}</>}

    {tab === "gate" && <><div className="infra-two-columns"><section className="infra-panel regression-config"><header><div><span>REGRESSION GATE</span><h2>候选版本不能悄悄变差</h2></div><b>{regression?.status === "passed" ? "PASS" : regression ? "BLOCK" : "WAITING"}</b></header><label>Baseline<select value={baselineId} onChange={(event) => setBaselineId(event.target.value)}><option value="">选择基线实验</option>{successfulExperiments.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label><label>Candidate<select value={candidateId} onChange={(event) => setCandidateId(event.target.value)}><option value="">选择候选实验</option>{successfulExperiments.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}</select></label><button className="primary-button" disabled={!baselineId || !candidateId || Boolean(busy)} onClick={() => void compareRuns()}>运行回归门禁</button>{regression && <div className="gate-checks">{regression.checks.map((check) => <article key={check.name}><i>{check.status === "passed" ? "✓" : "!"}</i><span><strong>{check.name}</strong><small>{check.rule}</small></span><b>{check.delta > 0 ? "+" : ""}{check.delta}</b></article>)}</div>}</section><section className="infra-panel performance-lab"><header><div><span>PERFORMANCE BENCHMARK</span><h2>FAISS 规模压测</h2></div><b>NO QUALITY CLAIM</b></header><p>生成固定 seed 的 1k / 10k 向量，对比 Flat 与 HNSW 的构建时间、P50/P95/P99、QPS 和 ANN Recall。</p><button className="secondary-button" disabled={Boolean(busy)} onClick={() => void runBenchmark()}>{busy === "benchmark" ? "入队中…" : "运行 1k / 10k 压测"}</button><div className="benchmark-list">{benchmarks.slice(0, 4).map((item) => <article key={item.id}><span><strong>{(item.config.sizes || []).map((value: number) => value.toLocaleString()).join(" / ")} vectors</strong><small>{item.config.dimension}d · seed {item.config.seed}</small></span><b className={`infra-status ${item.status}`}>{item.status}</b>{item.result.results?.filter((result) => result.status === "measured").map((result) => <em key={result.size}>{Number(result.size).toLocaleString()}: Flat P95 {result.flat?.p95_ms} ms · HNSW {result.hnsw?.p95_ms} ms · ANN R@10 {Math.round((result.hnsw?.ann_recall_10 || 0) * 100)}%</em>)}</article>)}</div></section></div>{regression && <section className={`regression-verdict ${regression.status}`}><i>{regression.status === "passed" ? "✓" : "!"}</i><div><span>RELEASE DECISION</span><h2>{regression.status === "passed" ? "候选配置通过门禁" : "候选配置被阻塞"}</h2><p>Evidence Recall 差值 95% CI：[{regression.confidence.evidence_recall_delta_95_ci.join(", ")}] · paired bootstrap {regression.confidence.samples} 次。</p></div></section>}</>}
  </div>;
}

function IndexStatusModal({ status, close, reindex }: { status: DocumentStatus; close: () => void; reindex: () => void }) {
  const labels: Record<string, string> = { ready: "可检索", lexical_ready: "关键词可检索，语义索引中", indexing: "正在索引", failed: "索引失败", missing: "本地副本缺失", stale: "待重新索引" };
  return <div className="modal-backdrop"><section className="index-status-modal"><header><div><span>索引详情</span><h2>{status.title}</h2><p>{status.original_name}</p></div><button onClick={close}>×</button></header><div className={`status-hero status-${status.status}`}><i /> <strong>{labels[status.status] || status.status}</strong><p>{status.status === "ready" ? "本地副本和检索索引均可使用。" : status.status === "missing" ? "KUN 资料库中的独立副本已不存在，当前索引不可继续核对原文。" : status.latest_job?.message || "查看下方详细状态。"}</p></div><dl><div><dt>KUN 本地副本</dt><dd>{status.library_copy_exists ? "存在" : "缺失"}</dd></div><div><dt>知识片段</dt><dd>{status.chunk_count} 个</dd></div><div><dt>语义向量</dt><dd>{status.embedding_count} 个</dd></div><div><dt>Embedding 模型</dt><dd>{status.embedding_model || "仅 BM25"}</dd></div><div><dt>最后更新</dt><dd>{formatDate(status.updated_at)}</dd></div><div><dt>副本位置</dt><dd title={status.library_path}>{status.library_path}</dd></div></dl><footer><button className="ghost-button" onClick={() => void navigator.clipboard.writeText(status.library_path)}>复制副本路径</button><button className="primary-button" disabled={!status.library_copy_exists || status.status === "indexing"} onClick={reindex}>重新建立索引</button></footer></section></div>;
}

function SettingsView({ providers, health, testProvider }: { providers: ProviderStatus[]; health: Health | null; testProvider: (provider: string) => Promise<void> }) {
  const [section, setSection] = useState<"models" | "storage" | "privacy">("models");
  const [storage, setStorage] = useState<StorageInfo | null>(null);
  const [privacy, setPrivacy] = useState<PrivacySettings | null>(null);
  const [busy, setBusy] = useState("");
  const [notice, setNotice] = useState("");
  async function loadStorage() {
    setStorage(await api<StorageInfo>("/api/settings/storage"));
  }
  async function loadPrivacy() {
    setPrivacy(await api<PrivacySettings>("/api/settings/privacy"));
  }
  useEffect(() => {
    if (section === "storage") void loadStorage().catch((error) => setNotice(error instanceof Error ? error.message : "读取存储信息失败"));
    if (section === "privacy") void loadPrivacy().catch((error) => setNotice(error instanceof Error ? error.message : "读取隐私设置失败"));
  }, [section]);
  async function createBackup() {
    setBusy("backup"); setNotice("");
    try {
      const created = await api<BackupItem>("/api/settings/backups", { method: "POST" });
      setNotice(`备份已创建：${created.id}`);
      await loadStorage();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "创建备份失败");
    } finally {
      setBusy("");
    }
  }
  async function restoreBackup(item: BackupItem) {
    const confirmation = window.prompt(`恢复会把当前资料、索引、对话和 Memory 替换为该备份内容。\n系统会先自动创建一份恢复前备份。\n\n请输入：恢复此备份`);
    if (confirmation !== "恢复此备份") return;
    setBusy(item.id); setNotice("");
    try {
      await api("/api/settings/backups/restore", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ backup_id: item.id, confirmation }),
      });
      setNotice("恢复完成。建议刷新页面重新读取资料、对话和 Memory。");
      await loadStorage();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "恢复备份失败");
    } finally {
      setBusy("");
    }
  }
  async function updatePrivacy(key: keyof PrivacySettings, value: boolean) {
    if (!privacy || key === "fixed_boundaries" || key === "sensitive_data_protection_enabled") return;
    setPrivacy({ ...privacy, [key]: value });
    setNotice("");
    try {
      setPrivacy(await api<PrivacySettings>("/api/settings/privacy", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ [key]: value }),
      }));
      setNotice("隐私设置已保存，并会立即影响后续 Agent 与 Tool 调用。");
    } catch (error) {
      await loadPrivacy();
      setNotice(error instanceof Error ? error.message : "隐私设置保存失败");
    }
  }
  return <div className="page settings-page">
    <PageHead eyebrow="SETTINGS" title="设置" description="模型连接、本地数据、备份与权限边界都在这里真实生效。" />
    <div className="settings-layout"><nav><button className={section === "models" ? "selected" : ""} onClick={() => setSection("models")}>模型与 API</button><button className={section === "storage" ? "selected" : ""} onClick={() => setSection("storage")}>存储与备份</button><button className={section === "privacy" ? "selected" : ""} onClick={() => setSection("privacy")}>隐私与权限</button></nav><section>
      {notice && <div className="settings-notice">{notice}</div>}
      {section === "models" && <>{providers.map((provider) => <div className="setting-section" key={provider.provider}><h3>{provider.capability === "chat" ? "对话模型" : provider.capability === "rerank" ? "Reranker 模型" : "Embedding 模型"}</h3><p>{provider.capability === "chat" ? "负责理解问题、规划 Tool 和生成回答。" : provider.capability === "rerank" ? "负责对融合候选重新排序，需要配置百炼工作空间 Rerank 地址。" : "负责将文档片段与问题转换为语义向量。"}</p><div className="model-row"><div className={`provider-logo ${provider.provider === "deepseek" ? "deepseek" : "aliyun"}`}>{provider.provider === "deepseek" ? "D" : "A"}</div><div><strong>{provider.label}</strong><small>{provider.model}</small></div><ProviderBadge provider={provider} /><button onClick={() => testProvider(provider.provider)} disabled={!provider.configured}>测试连接</button></div></div>)}
        <div className="setting-section inline-settings"><div><h3>本地服务</h3><p>FastAPI 仅监听 127.0.0.1</p></div><span className={health ? "connected" : "failed-state"}>{health ? "● 正常" : "● 未连接"}</span></div></>}
      {section === "storage" && (storage ? <div className="storage-settings">
        <div className="storage-hero"><div><span>本地数据总量</span><strong>{formatSize(storage.total_bytes)}</strong><p title={storage.data_dir}>{storage.data_dir}</p></div><div><button className="secondary-button" onClick={() => void api("/api/settings/storage/open", { method: "POST" })}>打开资料位置</button><button className="primary-button" disabled={busy === "backup"} onClick={() => void createBackup()}>{busy === "backup" ? "正在备份…" : "立即创建备份"}</button></div></div>
        <div className="storage-counts"><div><strong>{storage.counts.documents}</strong><span>份资料</span></div><div><strong>{storage.counts.chunks}</strong><span>个知识片段</span></div><div><strong>{storage.counts.conversations}</strong><span>段对话</span></div><div><strong>{storage.counts.memories}</strong><span>条 Memory</span></div></div>
        <section className="storage-breakdown"><h3>空间占用</h3>{[["library", "资料副本"], ["indexes", "索引"], ["database", "数据库"], ["backups", "备份"], ["staging", "待确认资料"], ["exports", "导出文件"]].map(([key, label]) => { const bytes = key === "database" ? storage.database_bytes : storage.folders[key] || 0; const width = storage.total_bytes ? Math.max(2, Math.round(bytes / storage.total_bytes * 100)) : 2; return <div key={key}><span>{label}</span><i><b style={{ width: `${width}%` }} /></i><strong>{formatSize(bytes)}</strong></div>; })}</section>
        <section className="backup-list"><header><div><h3>本机备份</h3><p>ZIP 包含数据库、资料副本、索引、Skills 和导出文件，不会上传云端。</p></div><span>{storage.backups.length} 份</span></header>{storage.backups.length ? storage.backups.map((item) => <article key={item.id}><div><strong>{item.reason === "pre-restore" ? "恢复前安全备份" : "手动备份"}</strong><p>{formatDate(item.created_at)} · {formatSize(item.size_bytes)}</p><small>{item.id}</small></div><a href={`${API_BASE}/api/settings/backups/${encodeURIComponent(item.id)}`}>下载</a><button disabled={busy === item.id || item.status !== "ready"} onClick={() => void restoreBackup(item)}>{busy === item.id ? "恢复中…" : "恢复"}</button></article>) : <div className="backup-empty">还没有备份。首次导入重要资料后，建议立即创建一份。</div>}</section>
        <div className="backup-warning"><b>备份也包含私人数据</b><p>请保存在受 Windows 账户保护的磁盘中，不建议放入公共网盘或共享文件夹。恢复前系统会自动制作安全备份。</p></div>
      </div> : <div className="settings-loading">正在统计本地数据…</div>)}
      {section === "privacy" && (privacy ? <div className="privacy-settings">
        <div className="privacy-summary"><span>本地优先</span><h2>默认保留在你的电脑上</h2><p>只有下列明确开启的能力，才会把完成任务所需的最小内容发送给对应模型服务。</p></div>
        <section><h3>数据出站控制</h3><SettingToggle title="自主联网搜索" text="问题包含“最新、今天、上网查”等意图时，允许 web.search 把查询发送到公网搜索服务。" checked={privacy.web_search_enabled} onChange={(value) => void updatePrivacy("web_search_enabled", value)} /><SettingToggle title="云端文档临时理解" text="导入文档时，允许将有限文本节选发送给 DeepSeek，用于生成待确认的标题、摘要和标签。" checked={privacy.cloud_document_analysis_enabled} onChange={(value) => void updatePrivacy("cloud_document_analysis_enabled", value)} /><SettingToggle title="云端图片理解与 OCR" text="导入图片时，允许将压缩副本发送给百炼视觉模型；关闭后不会执行云端 OCR 或画面描述。" checked={privacy.cloud_image_analysis_enabled} onChange={(value) => void updatePrivacy("cloud_image_analysis_enabled", value)} /></section>
        <section><h3>个人信息与 Memory</h3><SettingToggle title="从对话中生成 Memory 建议" text="识别姓名、地点、偏好、目标等稳定信息。非明确指令只生成候选，仍需你确认后才进入长期 Memory。" checked={privacy.memory_suggestions_enabled} onChange={(value) => void updatePrivacy("memory_suggestions_enabled", value)} /><SettingToggle title="敏感信息保护" text="密码、验证码、银行卡、手机号和精确住址不会自动保存为 Memory。该安全边界不能关闭。" checked={privacy.sensitive_data_protection_enabled} locked onChange={() => undefined} /></section>
        <section className="fixed-boundaries"><h3>固定权限边界</h3>{Object.values(privacy.fixed_boundaries).map((value) => <div key={value}><span>✓</span><p>{value}</p></div>)}</section>
      </div> : <div className="settings-loading">正在读取隐私边界…</div>)}
    </section></div>
  </div>;
}

function SettingToggle({ title, text, checked, locked = false, onChange }: { title: string; text: string; checked: boolean; locked?: boolean; onChange: (value: boolean) => void }) {
  return <label className={`setting-toggle ${locked ? "locked" : ""}`}><div><strong>{title}</strong><p>{text}</p></div><input type="checkbox" aria-label={title} checked={checked} disabled={locked} onChange={(event) => onChange(event.target.checked)} /><span aria-hidden="true" /></label>;
}

function ProviderBadge({ provider }: { provider: ProviderStatus }) {
  if (!provider.configured) return <span className="provider-state pending">待配置</span>;
  if (provider.connection_status === "connected") return <span className="provider-state connected">✓ 已连接</span>;
  if (provider.connection_status === "failed") return <span className="provider-state failed-state">连接失败</span>;
  return <span className="provider-state pending">待验证</span>;
}

function EmptyFeature({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action: string }) {
  return <div className="page"><PageHead eyebrow={eyebrow} title={title} description={description} /><EmptyState title={action} text="该能力会在真实后端接通并验证后开放。" /></div>;
}

function EmptyState({ title, text, action }: { title: string; text: string; action?: ReactNode }) {
  return <div className="empty-state"><div className="kun-orb small">K</div><h3>{title}</h3><p>{text}</p>{action}</div>;
}

function UploadReview({ staged, imported, status, progress, error, close, confirm, update }: {
  staged: StagedFile[];
  imported: boolean;
  status: string;
  progress: number | null;
  error: string;
  close: () => void;
  confirm: () => void;
  update: (id: string, patch: Partial<StagedFile>) => void;
}) {
  const [selectedId, setSelectedId] = useState<string>("");
  useEffect(() => { if (staged.length && !staged.some((item) => item.id === selectedId)) setSelectedId(staged[0].id); }, [staged, selectedId]);
  const selected = staged.find((item) => item.id === selectedId) || staged[0];
  return <div className="modal-backdrop"><section className="upload-modal"><header><div><span>待确认资料</span><h2>{status || `${staged.length} 个文件已完成临时理解`}</h2><p>确认前不会进入正式知识库；云端模型只接收用于生成元数据的文本节选。</p>{progress !== null && status && <div className="index-progress" aria-label={`索引进度 ${progress}%`}><i style={{ width: `${progress}%` }} /><span>{progress}%</span></div>}</div><button onClick={close}>×</button></header>
    {error && <div className="modal-error">{error}</div>}
    {imported ? <div className="success-state"><div>✓</div><h3>已建立真实索引</h3><p>现在可以在对话中询问这批资料，并点击引用核对原文。</p><button className="primary-button" onClick={close}>开始提问</button></div>
        : status && !staged.length ? <div className="loading-state"><span className="pulse-dot" /><h3>{status}</h3><p>图片会先完成视觉理解和 OCR；大文件解析可能需要一些时间，请保持页面打开。</p></div>
      : <><div className="review-body"><aside>{staged.map((file) => <button className={file.id === selected?.id ? "selected" : ""} key={file.id} onClick={() => setSelectedId(file.id)}><b>{file.file_type.slice(0, 2).toUpperCase()}</b><span><strong>{file.original_name}</strong><small>{formatSize(file.size_bytes)} · {file.sections} 个结构单元</small></span><i>✓</i></button>)}</aside>{selected && <div className="review-form"><label>通俗标题<input value={selected.title} onChange={(event) => update(selected.id, { title: event.target.value })} /></label><label>简短摘要<textarea value={selected.summary} onChange={(event) => update(selected.id, { summary: event.target.value })} /></label><label>标签<input value={selected.tags.join("，")} onChange={(event) => update(selected.id, { tags: event.target.value.split(/[，,]/).map((item) => item.trim()).filter(Boolean) })} /></label><div className="parse-summary"><span>✓</span><div><strong>{selected.metadata_source === "qwen_vision+deepseek" ? "视觉模型已完成 OCR 和画面理解" : selected.metadata_source === "deepseek" ? "DeepSeek 已完成临时理解" : "当前使用本地规则生成元数据"}</strong><p>{selected.metadata_source === "qwen_vision+deepseek" ? "标题、摘要和标签已根据图片文字与画面生成；确认后直接建立语义索引。" : "请检查标题、摘要和标签；确认后才会切分、Embedding 并建立索引。"}</p></div></div></div>}</div><footer><button className="ghost-button" onClick={close}>稍后处理</button><div><span>{status || `将确认 ${staged.length} 个文件`}</span><button className="primary-button" onClick={confirm} disabled={Boolean(status) || !staged.length}>确认并建立索引</button></div></footer></>}
  </section></div>;
}

function Onboarding({ step, setStep, finish, providers, testProvider, backendOnline }: {
  step: number;
  setStep: (step: number) => void;
  finish: () => void;
  providers: ProviderStatus[];
  testProvider: (provider: string) => Promise<void>;
  backendOnline: boolean;
}) {
  const [libraryName, setLibraryName] = useState("KUN Library（默认位置）");
  const directoryInput = useRef<HTMLInputElement>(null);
  async function chooseLibraryFolder() {
    const picker = window as Window & { showDirectoryPicker?: () => Promise<{ name: string }> };
    if (!picker.showDirectoryPicker) return directoryInput.current?.click();
    try {
      const directory = await picker.showDirectoryPicker();
      setLibraryName(directory.name);
    } catch (error) {
      if (!(error instanceof DOMException && error.name === "AbortError")) setLibraryName("文件夹选择失败，请重试");
    }
  }
  function fallbackFolder(event: ChangeEvent<HTMLInputElement>) {
    const first = event.target.files?.[0];
    if (first) setLibraryName(first.webkitRelativePath.split("/")[0] || "已选择本地资料库");
    event.target.value = "";
  }
  const steps = [
    { n: "01", title: "欢迎认识坤坤", text: "你的个人私域知识智能体。文件、索引和 Memory 默认保存在这台电脑上。", visual: <div className="welcome-visual"><div className="kun-orb">K</div><i className="orbit one" /><i className="orbit two" /></div> },
    { n: "02", title: "选择本地资料库", text: "KUN 会复制一份资料用于稳定引用，原始文件不会被修改。", visual: <div className="folder-visual"><span>▰</span><p>{libraryName}</p><button onClick={chooseLibraryFolder}>更改位置</button></div> },
    { n: "03", title: "验证模型能力", text: "只有真实测试成功后才显示绿色“已连接”。", visual: <div className="setup-options">{providers.map((provider) => <div key={provider.provider}><b>{provider.provider === "deepseek" ? "D" : "A"}</b><span>{provider.capability === "chat" ? "对话模型" : provider.capability === "rerank" ? "Reranker" : "Embedding"}<strong>{provider.label} · {provider.model}</strong></span><ProviderBadge provider={provider} /><button onClick={() => testProvider(provider.provider)} disabled={!backendOnline || !provider.configured}>测试</button></div>)}</div> },
    { n: "04", title: "创建第一个知识空间", text: "默认空间会由本地后端创建，之后可以继续增加和隔离不同主题。", visual: <div className="space-name"><label>知识空间名称<input value="AI Agent 学习" readOnly /></label><div><span>本地保存</span><span>独立索引</span><span>可追溯引用</span></div></div> },
  ];
  const current = steps[step];
  return <div className="modal-backdrop onboarding"><input ref={directoryInput} type="file" hidden onChange={fallbackFolder} {...({ webkitdirectory: "", directory: "" } as any)} /><section className="onboarding-card"><div className="onboarding-side"><div className="mini-brand"><b>K</b><span>KUN</span></div><div>{steps.map((item, index) => <span key={item.n} className={index === step ? "active" : index < step ? "done" : ""}>{index < step ? "✓" : item.n}<small>{item.title}</small></span>)}</div><p>Knowledge · Understanding · Navigation</p></div><div className="onboarding-content"><span className="step-label">步骤 {step + 1} / 4</span>{current.visual}<h1>{current.title}</h1><p>{current.text}</p><div className="onboarding-actions"><button disabled={step === 0} onClick={() => setStep(step - 1)}>上一步</button><button className="primary-button" onClick={() => step === 3 ? finish() : setStep(step + 1)}>{step === 3 ? "开始使用 KUN" : "继续"}　→</button></div></div></section></div>;
}
