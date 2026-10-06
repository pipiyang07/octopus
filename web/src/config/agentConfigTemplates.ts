export type AgentType = "codex" | "claude" | "pi" | "dsh";

export type ReasoningLevel =
    | "none"
    | "minimal"
    | "low"
    | "medium"
    | "high"
    | "xhigh"
    | "max"
    | "ultra";

export interface AgentModelConfig {
    id: string;
    context1M: boolean;
}

export interface AgentConfigInput {
    agent: AgentType;
    baseURL: string;
    apiKey: string;
    models: AgentModelConfig[];
    defaultModel: string;
    defaultReasoningLevel: ReasoningLevel;
}

export interface AgentConfigFile {
    key: string;
    fileName: string;
    title: string;
    macOSPath: string;
    windowsPath: string;
    language: "toml" | "json" | "yaml";
    content: string;
    mergeFragment?: string;
}

const CODEX_CATALOG_FILENAME = "octopus-model-catalog.json";
const DEFAULT_CONTEXT_WINDOW = 128_000;
const ONE_MEBI_CONTEXT_WINDOW = 1_048_576;
const CODEX_BASE_INSTRUCTIONS =
    "You are Codex, a coding agent. You and the user share the same workspace and collaborate to achieve the user's goals.";

const REASONING_DESCRIPTIONS: Record<ReasoningLevel, string> = {
    none: "Disable Thinking",
    minimal: "Minimal reasoning",
    low: "Fast responses with lighter reasoning",
    medium: "Balances speed and reasoning depth for everyday tasks",
    high: "Greater reasoning depth for complex problems",
    xhigh: "Extra high reasoning depth for complex problems",
    max: "Maximum reasoning depth for the hardest problems",
    ultra: "Ultra reasoning depth",
};

type PiThinkingLevel = "off" | Exclude<ReasoningLevel, "none" | "ultra">;

const PI_THINKING_LEVELS: PiThinkingLevel[] = [
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
];

function normalizeOrigin(baseURL: string): string {
    return baseURL.trim().replace(/\/+$/, "");
}

function joinOpenAIV1(baseURL: string): string {
    const origin = normalizeOrigin(baseURL);
    return /\/v1$/.test(origin) ? origin : `${origin}/v1`;
}

function tomlString(value: string): string {
    return JSON.stringify(value);
}

function contextWindow(context1M: boolean): number {
    return context1M ? ONE_MEBI_CONTEXT_WINDOW : DEFAULT_CONTEXT_WINDOW;
}

function findDefaultModel(input: AgentConfigInput): AgentModelConfig {
    const found = input.models.find((model) => model.id === input.defaultModel);
    return found ?? input.models[0];
}

function claudeModelID(model: AgentModelConfig): string {
    return model.context1M ? `${model.id}[1m]` : model.id;
}

function claudeEffort(level: ReasoningLevel): string {
    if (level === "minimal") return "low";
    if (level === "none" || level === "ultra") return "high";
    return level;
}

function dshEffort(level: ReasoningLevel): string {
    if (level === "none") return "off";
    if (level === "minimal") return "minimal";
    if (level === "ultra") return "max";
    return level;
}

function prettyJSON(value: unknown): string {
    return `${JSON.stringify(value, null, 2)}\n`;
}

export function generateCodexConfigFiles(input: AgentConfigInput): AgentConfigFile[] {
    const defaultModel = findDefaultModel(input);
    const providerBaseURL = joinOpenAIV1(input.baseURL);
    const defaultContext = contextWindow(defaultModel?.context1M ?? false);

    const config = [
        `model_provider = "octopus"`,
        `model = ${tomlString(defaultModel?.id ?? "")}`,
        `model_reasoning_effort = ${tomlString(input.defaultReasoningLevel)}`,
        `model_context_window = ${defaultContext}`,
        `disable_response_storage = true`,
        `model_catalog_json = ${tomlString(CODEX_CATALOG_FILENAME)}`,
        ``,
        `[model_providers.octopus]`,
        `name = "Octopus"`,
        `base_url = ${tomlString(providerBaseURL)}`,
        `wire_api = "responses"`,
        `experimental_bearer_token = ${tomlString(input.apiKey)}`,
        `requires_openai_auth = false`,
        ``,
    ].join("\n");

    const catalog = {
        models: input.models.map((model, index) => ({
            slug: model.id,
            display_name: model.id,
            description: `${model.id} routed by Octopus`,
            base_instructions: CODEX_BASE_INSTRUCTIONS,
            default_reasoning_level: input.defaultReasoningLevel,
            supported_reasoning_levels: (
                Object.keys(REASONING_DESCRIPTIONS) as ReasoningLevel[]
            ).map((effort) => ({
                effort,
                description: REASONING_DESCRIPTIONS[effort],
            })),
            shell_type: "shell_command",
            visibility: "list",
            supported_in_api: true,
            priority: 1000 + index,
            supports_reasoning_summaries: true,
            default_reasoning_summary: "none",
            support_verbosity: false,
            truncation_policy: { mode: "bytes", limit: 10_000 },
            supports_parallel_tool_calls: false,
            supports_image_detail_original: false,
            context_window: contextWindow(model.context1M),
            max_context_window: contextWindow(model.context1M),
            effective_context_window_percent: 95,
            experimental_supported_tools: [],
            input_modalities: ["text"],
            supports_search_tool: false,
        })),
    };

    return [
        {
            key: "codex-config",
            fileName: "config.toml",
            title: "Codex config.toml",
            macOSPath: "~/.codex/config.toml",
            windowsPath: "%USERPROFILE%\\.codex\\config.toml",
            language: "toml",
            content: config,
        },
        {
            key: "codex-catalog",
            fileName: CODEX_CATALOG_FILENAME,
            title: "Codex model catalog",
            macOSPath: `~/.codex/${CODEX_CATALOG_FILENAME}`,
            windowsPath: `%USERPROFILE%\\.codex\\${CODEX_CATALOG_FILENAME}`,
            language: "json",
            content: prettyJSON(catalog),
        },
    ];
}

export function generateClaudeConfigFiles(input: AgentConfigInput): AgentConfigFile[] {
    const defaultModel = findDefaultModel(input);
    const model = claudeModelID(defaultModel);
    const env: Record<string, string> = {
        ANTHROPIC_BASE_URL: normalizeOrigin(input.baseURL),
        ANTHROPIC_AUTH_TOKEN: input.apiKey,
        ANTHROPIC_MODEL: model,
        ANTHROPIC_DEFAULT_HAIKU_MODEL: model,
        ANTHROPIC_DEFAULT_SONNET_MODEL: model,
        ANTHROPIC_DEFAULT_OPUS_MODEL: model,
    };

    if (input.defaultReasoningLevel === "none") {
        env.MAX_THINKING_TOKENS = "0";
    } else {
        env.CLAUDE_CODE_EFFORT_LEVEL = claudeEffort(input.defaultReasoningLevel);
    }

    const settings = {
        env,
        availableModels: input.models.map(claudeModelID),
        modelPicker: {
            options: input.models.map((model) => ({
                model: claudeModelID(model),
                label: model.context1M ? `${model.id} (1M)` : model.id,
                description: model.context1M
                    ? "1M context model routed by Octopus"
                    : "Model routed by Octopus",
            })),
            replaceBuiltInOptions: true,
        },
    };
    const full = prettyJSON(settings);
    const mergeFragment = full;

    return [
        {
            key: "claude-settings",
            fileName: "settings.json",
            title: "Claude Code settings.json",
            macOSPath: "~/.claude/settings.json",
            windowsPath: "%USERPROFILE%\\.claude\\settings.json",
            language: "json",
            content: full,
            mergeFragment,
        },
    ];
}

export function generatePiConfigFiles(input: AgentConfigInput): AgentConfigFile[] {
    const provider = {
        octopus: {
            name: "Octopus",
            baseUrl: joinOpenAIV1(input.baseURL),
            api: "openai-completions",
            apiKey: input.apiKey,
            models: input.models.map((model) => ({
                id: model.id,
                name: model.id,
                reasoning: true,
                input: ["text"],
                contextWindow: contextWindow(model.context1M),
                maxTokens: 32_768,
                thinkingLevelMap: Object.fromEntries(
                    PI_THINKING_LEVELS.map((level) => [
                        level,
                        level === "off" ? "none" : level,
                    ]),
                ),
            })),
        },
    };

    const full = prettyJSON({ providers: provider });
    const mergeFragment = prettyJSON(provider.octopus);

    return [
        {
            key: "pi-models",
            fileName: "models.json",
            title: "Pi models.json",
            macOSPath: "~/.pi/agent/models.json",
            windowsPath: "%USERPROFILE%\\.pi\\agent\\models.json",
            language: "json",
            content: full,
            mergeFragment,
        },
    ];
}

export function generateDSHConfigFiles(input: AgentConfigInput): AgentConfigFile[] {
    const modelLines = input.models.map((model) => [
        `          - id: ${tomlString(model.id)}`,
        `            contextWindow: ${contextWindow(model.context1M)}`,
        `            maxTokens: 32768`,
        `            input: [text]`,
        `            reasoningEfforts:`,
        `              off:`,
        `              minimal: minimal`,
        `              low: low`,
        `              medium: medium`,
        `              high: high`,
        `              xhigh: xhigh`,
        `              max: max`,
    ].join("\n")).join("\n");

    const defaultReasoning = input.defaultReasoningLevel === "none"
        ? ""
        : `        reasoning: ${dshEffort(input.defaultReasoningLevel)}\n`;
    const providerPatch = [
        `- id: llm-pi-ai`,
        `  config:`,
        `    providers:`,
        `      octopus:`,
        `        apiKeyEnv: OCTOPUS_API_KEY`,
        `        api: openai-responses`,
        `        baseURL: ${tomlString(joinOpenAIV1(input.baseURL))}`,
        defaultReasoning,
        `        models:`,
        modelLines,
        ``,
    ].join("\n");

    const credentials = [
        `version: 1`,
        `refs:`,
        `  OCTOPUS_API_KEY: ${tomlString(input.apiKey)}`,
        ``,
    ].join("\n");

    return [
        {
            key: "dsh-provider",
            fileName: "cordis.patch.yml",
            title: "DSH provider patch",
            macOSPath: "~/.dsh/profiles/web/cordis.patch.yml",
            windowsPath: "%USERPROFILE%\\.dsh\\profiles\\web\\cordis.patch.yml",
            language: "yaml",
            content: providerPatch,
        },
        {
            key: "dsh-credentials",
            fileName: ".credentials.yaml",
            title: "DSH credentials",
            macOSPath: "~/.dsh/.credentials.yaml",
            windowsPath: "%USERPROFILE%\\.dsh\\.credentials.yaml",
            language: "yaml",
            content: credentials,
        },
    ];
}

export function generateAgentConfigFiles(input: AgentConfigInput): AgentConfigFile[] {
    if (input.agent === "claude") return generateClaudeConfigFiles(input);
    if (input.agent === "pi") return generatePiConfigFiles(input);
    if (input.agent === "dsh") return generateDSHConfigFiles(input);
    return generateCodexConfigFiles(input);
}
