'use client';

import { useMemo, useState, useSyncExternalStore } from 'react';
import { useTranslations } from 'next-intl';
import { AlertTriangle, Copy, FolderCog, RotateCcw } from 'lucide-react';
import { useGroupList } from '@/api/endpoints/group';
import { useAPIKeyList } from '@/api/endpoints/apikey';
import { PageWrapper } from '@/components/common/PageWrapper';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { toast } from '@/components/common/Toast';
import {
    generateAgentConfigFiles,
    type AgentType,
    type ReasoningLevel,
} from '@/config/agentConfigTemplates';

const AGENTS: Array<{ id: AgentType; label: string }> = [
    { id: 'codex', label: 'Codex CLI' },
    { id: 'claude', label: 'Claude Code' },
    { id: 'pi', label: 'Pi' },
    { id: 'dsh', label: 'DSH' },
];

const REASONING_LEVELS: ReasoningLevel[] = [
    'none',
    'minimal',
    'low',
    'medium',
    'high',
    'xhigh',
    'max',
    'ultra',
];

function maskAPIKey(key: string): string {
    if (key.length <= 12) return key;
    return `${key.slice(0, 10)}...${key.slice(-4)}`;
}

const subscribeToNothing = () => () => {};
const getBrowserOrigin = () => window.location.origin;
const getServerOrigin = () => 'http://127.0.0.1:8080';

export function AgentConfig() {
    const t = useTranslations('agentConfig');
    const { data: groups } = useGroupList();
    const { data: apiKeys } = useAPIKeyList();

    const [agent, setAgent] = useState<AgentType>('codex');
    const browserOrigin = useSyncExternalStore(subscribeToNothing, getBrowserOrigin, getServerOrigin);
    const [baseURLOverride, setBaseURLOverride] = useState<string | null>(null);
    const [apiKeyIDOverride, setAPIKeyID] = useState<string | null>(null);
    const [selectedModelsOverride, setSelectedModelsOverride] = useState<string[] | null>(null);
    const [oneMModels, setOneMModels] = useState<Record<string, boolean>>({});
    const [defaultModelOverride, setDefaultModel] = useState<string | null>(null);
    const [reasoningLevel, setReasoningLevel] = useState<ReasoningLevel>('medium');
    const [activeFileKeyOverride, setActiveFileKey] = useState<string | null>(null);
    const [viewModeOverride, setViewMode] = useState<'full' | 'merge' | null>(null);
    const [contentOverrides, setContentOverrides] = useState<Record<string, string>>({});
    const baseURL = baseURLOverride ?? browserOrigin;

    const enabledAPIKeys = useMemo(
        () => (apiKeys ?? []).filter((key) => key.enabled),
        [apiKeys],
    );

    const apiKeyID = apiKeyIDOverride ?? (enabledAPIKeys[0] ? String(enabledAPIKeys[0].id) : '');
    const selectedAPIKey = useMemo(
        () => enabledAPIKeys.find((key) => String(key.id) === apiKeyID),
        [enabledAPIKeys, apiKeyID],
    );

    const availableModels = useMemo(() => {
        const models = new Set((groups ?? []).map((group) => group.name.trim()).filter(Boolean));
        if (!selectedAPIKey?.supported_models?.trim()) {
            return [...models].sort((a, b) => a.localeCompare(b));
        }
        const allowed = new Set(
            selectedAPIKey.supported_models
                .split(',')
                .map((model) => model.trim())
                .filter(Boolean),
        );
        return [...models]
            .filter((model) => allowed.has(model))
            .sort((a, b) => a.localeCompare(b));
    }, [groups, selectedAPIKey]);

    const selectedModels = useMemo(
        () => (selectedModelsOverride ?? availableModels)
            .filter((model) => availableModels.includes(model))
            .sort((a, b) => a.localeCompare(b)),
        [availableModels, selectedModelsOverride],
    );
    const defaultModel = defaultModelOverride && selectedModels.includes(defaultModelOverride)
        ? defaultModelOverride
        : selectedModels[0] ?? '';

    const modelConfigs = useMemo(
        () => selectedModels.map((id) => ({ id, context1M: oneMModels[id] === true })),
        [oneMModels, selectedModels],
    );

    const files = useMemo(() => {
        if (!selectedAPIKey || modelConfigs.length === 0 || !defaultModel) return [];
        return generateAgentConfigFiles({
            agent,
            baseURL,
            apiKey: selectedAPIKey.api_key,
            models: modelConfigs,
            defaultModel,
            defaultReasoningLevel: reasoningLevel,
        });
    }, [agent, baseURL, defaultModel, modelConfigs, reasoningLevel, selectedAPIKey]);

    const activeFile = files.find((file) => file.key === activeFileKeyOverride) ?? files[0];
    const supportsMerge = Boolean(activeFile?.mergeFragment);
    const viewMode = supportsMerge ? (viewModeOverride ?? 'merge') : 'full';

    const storageKey = activeFile
        ? `${agent}:${activeFile.key}:${viewMode}`
        : `${agent}:none:${viewMode}`;
    const generatedContent = activeFile
        ? viewMode === 'merge' && activeFile.mergeFragment
            ? activeFile.mergeFragment
            : activeFile.content
        : '';
    const currentContent = contentOverrides[storageKey] ?? generatedContent;
    const edited = contentOverrides[storageKey] !== undefined;

    const toggleModel = (model: string) => {
        const next = selectedModels.includes(model)
            ? selectedModels.filter((item) => item !== model)
            : [...selectedModels, model].sort((a, b) => a.localeCompare(b));
        setSelectedModelsOverride(next);
    };

    const toggleOneM = (model: string, checked: boolean) => {
        setOneMModels((current) => ({ ...current, [model]: checked }));
    };

    const resetCurrentFile = () => {
        setContentOverrides((current) => {
            const next = { ...current };
            delete next[storageKey];
            return next;
        });
    };

    const copyText = async (text: string, successMessage: string) => {
        try {
            await navigator.clipboard.writeText(text);
            toast.success(successMessage);
        } catch (error) {
            const description = error instanceof Error ? error.message : String(error);
            toast.error(t('copyContent'), { description });
        }
    };

    return (
        <div className="h-full min-h-0 overflow-y-auto overscroll-contain rounded-t-3xl">
            <PageWrapper className="grid grid-cols-1 gap-4 pb-6 lg:grid-cols-[minmax(320px,380px)_minmax(0,1fr)]">
                <Card className="min-w-0">
                    <CardHeader>
                        <CardTitle>{t('title')}</CardTitle>
                        <CardDescription>{t('description')}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="space-y-2">
                            <label className="text-sm font-medium text-card-foreground" htmlFor="agent-config-agent">
                                {t('agent')}
                            </label>
                            <Select
                                value={agent}
                                onValueChange={(value) => {
                                    setAgent(value as AgentType);
                                    setContentOverrides({});
                                }}
                            >
                                <SelectTrigger id="agent-config-agent" className="w-full">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    {AGENTS.map((item) => (
                                        <SelectItem key={item.id} value={item.id}>
                                            {item.label}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium text-card-foreground" htmlFor="agent-config-base-url">
                                {t('baseURL')}
                            </label>
                            <Input
                                id="agent-config-base-url"
                                value={baseURL}
                                onChange={(event) => setBaseURLOverride(event.target.value)}
                                placeholder="http://127.0.0.1:8080"
                            />
                        </div>

                        <div className="space-y-2">
                            <label className="text-sm font-medium text-card-foreground" htmlFor="agent-config-api-key">
                                {t('apiKey')}
                            </label>
                            <Select value={apiKeyID} onValueChange={setAPIKeyID}>
                                <SelectTrigger id="agent-config-api-key" className="w-full">
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    {enabledAPIKeys.map((key) => (
                                        <SelectItem key={key.id} value={String(key.id)}>
                                            {key.name} · {maskAPIKey(key.api_key)}
                                        </SelectItem>
                                    ))}
                                </SelectContent>
                            </Select>
                        </div>

                        <div className="space-y-2">
                            <div className="flex flex-wrap items-center justify-between gap-2">
                                <label className="text-sm font-medium text-card-foreground">
                                    {t('models')}
                                </label>
                                <span className="text-xs text-muted-foreground">
                                    {selectedModels.length}/{availableModels.length}
                                </span>
                            </div>
                            {availableModels.length > 0 ? (
                                <div className="flex max-h-40 flex-wrap gap-1.5 overflow-y-auto rounded-xl border border-border bg-muted/30 p-2.5">
                                    {availableModels.map((model) => {
                                        const selected = selectedModels.includes(model);
                                        return (
                                            <Badge
                                                key={model}
                                                variant={selected ? 'default' : 'secondary'}
                                                className="cursor-pointer select-none"
                                                onClick={() => toggleModel(model)}
                                            >
                                                {model}
                                            </Badge>
                                        );
                                    })}
                                </div>
                            ) : (
                                <p className="rounded-xl border border-border bg-muted/30 px-3 py-2 text-sm text-muted-foreground">
                                    {selectedAPIKey ? t('noModels') : t('noAPIKeys')}
                                </p>
                            )}
                        </div>

                        {selectedModels.length > 0 ? (
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-card-foreground">
                                    {t('selectedModels')}
                                </label>
                                <p className="text-xs text-muted-foreground">{t('selectHint')}</p>
                                <div className="space-y-2">
                                    {selectedModels.map((model) => (
                                        <div
                                            key={model}
                                            className="flex items-center justify-between gap-3 rounded-xl border border-border bg-muted/20 px-3 py-2"
                                        >
                                            <span className="min-w-0 truncate text-sm">{model}</span>
                                            <label className="flex items-center gap-2 text-xs text-muted-foreground">
                                                {t('context1M')}
                                                <Switch
                                                    checked={oneMModels[model] === true}
                                                    onCheckedChange={(checked) => toggleOneM(model, checked)}
                                                />
                                            </label>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        ) : null}

                        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-card-foreground" htmlFor="agent-config-default-model">
                                    {t('defaultModel')}
                                </label>
                                <Select value={defaultModel} onValueChange={setDefaultModel}>
                                    <SelectTrigger id="agent-config-default-model" className="w-full">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {selectedModels.map((model) => (
                                            <SelectItem key={model} value={model}>
                                                {model}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>
                            <div className="space-y-2">
                                <label className="text-sm font-medium text-card-foreground" htmlFor="agent-config-reasoning">
                                    {t('reasoning')}
                                </label>
                                <Select
                                    value={reasoningLevel}
                                    onValueChange={(value) => setReasoningLevel(value as ReasoningLevel)}
                                >
                                    <SelectTrigger id="agent-config-reasoning" className="w-full">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        {REASONING_LEVELS.map((level) => (
                                            <SelectItem key={level} value={level}>
                                                {level}
                                            </SelectItem>
                                        ))}
                                    </SelectContent>
                                </Select>
                            </div>
                        </div>

                        <div className="flex items-start gap-2 rounded-xl border border-border bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
                            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                            <span>{t('security')}</span>
                        </div>
                    </CardContent>
                </Card>

                <Card className="min-w-0">
                    <CardHeader>
                        <CardTitle>{t('output')}</CardTitle>
                        <CardDescription>{activeFile?.title ?? t('noModels')}</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                        {files.length > 0 && activeFile ? (
                            <>
                                <div className="flex flex-wrap items-center gap-2">
                                    {files.map((file) => (
                                        <Button
                                            key={file.key}
                                            type="button"
                                            size="sm"
                                            variant={file.key === activeFile.key ? 'secondary' : 'ghost'}
                                            onClick={() => setActiveFileKey(file.key)}
                                        >
                                            {file.fileName}
                                        </Button>
                                    ))}
                                </div>

                                <div className="grid gap-2 rounded-xl border border-border bg-muted/20 p-3 text-xs text-muted-foreground sm:grid-cols-2">
                                    <div className="flex items-center justify-between gap-2">
                                        <span className="truncate">{t('pathMacOS')}: {activeFile.macOSPath}</span>
                                    </div>
                                    <div className="flex items-center justify-between gap-2">
                                        <span className="truncate">{t('pathWindows')}: {activeFile.windowsPath}</span>
                                    </div>
                                </div>

                                <div className="flex flex-wrap items-center justify-between gap-2">
                                    <div className="flex items-center gap-1">
                                        {supportsMerge ? (
                                            <>
                                                <Button
                                                    type="button"
                                                    size="sm"
                                                    variant={viewMode === 'merge' ? 'secondary' : 'ghost'}
                                                    onClick={() => setViewMode('merge')}
                                                >
                                                    {t('mergeFragment')}
                                                </Button>
                                                <Button
                                                    type="button"
                                                    size="sm"
                                                    variant={viewMode === 'full' ? 'secondary' : 'ghost'}
                                                    onClick={() => setViewMode('full')}
                                                >
                                                    {t('fullFile')}
                                                </Button>
                                            </>
                                        ) : null}
                                    </div>
                                    <div className="flex items-center gap-2">
                                        {edited ? (
                                            <>
                                                <Badge variant="outline">{t('edited')}</Badge>
                                                <Button type="button" size="sm" variant="ghost" onClick={resetCurrentFile}>
                                                    <RotateCcw className="size-4" />
                                                    {t('reset')}
                                                </Button>
                                            </>
                                        ) : null}
                                    </div>
                                </div>

                                <div className="flex flex-wrap items-center gap-2">
                                    <Button
                                        type="button"
                                        size="sm"
                                        disabled={!currentContent}
                                        onClick={() => void copyText(currentContent, t('copyContent'))}
                                    >
                                        <Copy className="size-4" />
                                        {t('copyContent')}
                                    </Button>
                                    <Button
                                        type="button"
                                        size="sm"
                                        variant="outline"
                                        onClick={() => void copyText(activeFile.macOSPath, t('copyPath'))}
                                    >
                                        <FolderCog className="size-4" />
                                        {t('copyPath')}
                                    </Button>
                                </div>

                                <textarea
                                    value={currentContent}
                                    onChange={(event) => setContentOverrides((current) => ({
                                        ...current,
                                        [storageKey]: event.target.value,
                                    }))}
                                    spellCheck={false}
                                    className="min-h-[420px] w-full rounded-xl border border-border bg-background p-4 font-mono text-xs leading-relaxed text-foreground outline-none focus-visible:ring-2 focus-visible:ring-ring"
                                />
                            </>
                        ) : (
                            <p className="rounded-xl border border-border bg-muted/20 px-3 py-2 text-sm text-muted-foreground">
                                {selectedAPIKey ? t('noModels') : t('noAPIKeys')}
                            </p>
                        )}
                    </CardContent>
                </Card>
            </PageWrapper>
        </div>
    );
}
