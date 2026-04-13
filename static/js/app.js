/* ============================================
   智能简历优化与模拟面试助手 - 前端逻辑
   阶段 3：新增简历分析功能
   ============================================ */

const API_BASE = window.location.pathname.startsWith('/resumeagent') ? '/resumeagent' : '';

// --- Session 管理 ---
let sessionId = localStorage.getItem('agent_session_id') || '';
if (!sessionId) {
    sessionId = generateSessionId();
    localStorage.setItem('agent_session_id', sessionId);
}

function generateSessionId() {
    return 'sess_' + Date.now().toString(36) + '_' + Math.random().toString(36).substring(2, 8);
}

function resetSession() {
    fetch(`${API_BASE}/agent/session/${sessionId}`, { method: 'DELETE' }).catch(() => {});
    sessionId = generateSessionId();
    localStorage.setItem('agent_session_id', sessionId);
    chatMessages.innerHTML = '';
    isInInterviewMode = false;
    messageInput.placeholder = '输入你的问题…';
    showWelcome();
    loadSessionList();
}

function showWelcome() {
    const welcome = document.createElement('div');
    welcome.className = 'message welcome';
    welcome.innerHTML = `<div class="message-content">
        <p>👋 你好！我是智能简历优化与模拟面试助手。</p>
        <p>你可以：</p>
        <ul>
            <li>📤 上传 <strong>PDF / 图片 / 文本</strong> 文件到知识库</li>
            <li>💬 基于已上传的知识进行问答</li>
            <li>🖼️ 直接发送图片进行即时问答</li>
            <li>🌐 询问实时信息（自动网络搜索）</li>
            <li>🎯 <strong>岗位分析</strong> — 上传或粘贴JD，获取技术要求解读+简历建议</li>
            <li>📋 <strong>简历分析</strong> — 上传或粘贴简历，获取专业评估报告</li>
        </ul>
        <p style="margin-top:8px;font-size:12px;color:#999;">会话 ID: ${escapeHtml(sessionId)}</p>
    </div>`;
    chatMessages.appendChild(welcome);
}

// --- DOM 元素 ---
const chatMessages = document.getElementById('chat-messages');
const messageInput = document.getElementById('message-input');
const btnSend = document.getElementById('btn-send');
const btnUpload = document.getElementById('btn-upload-kb');
const fileInput = document.getElementById('file-input');
const btnImage = document.getElementById('btn-image');
const imageInput = document.getElementById('image-input');
const imagePreview = document.getElementById('image-preview');
const previewImg = document.getElementById('preview-img');
const clearImageBtn = document.getElementById('clear-image');
const sessionList = document.getElementById('session-list');
const btnRefreshSessions = document.getElementById('btn-refresh-sessions');
const statusIndicator = document.getElementById('status-indicator');
const uploadModal = document.getElementById('upload-modal');
const progressFill = document.getElementById('progress-fill');
const uploadStatusText = document.getElementById('upload-status-text');
const btnCloseModal = document.getElementById('btn-close-modal');
const btnNewSession = document.getElementById('btn-new-session');

// 简历分析相关 DOM
const btnResume = document.getElementById('btn-resume');
const btnInterview = document.getElementById('btn-interview');
const resumeModal = document.getElementById('resume-modal');
const btnCloseResume = document.getElementById('btn-close-resume');
const resumeTabs = document.querySelectorAll('.resume-tab');
const resumeTabUpload = document.getElementById('resume-tab-upload');
const resumeTabPaste = document.getElementById('resume-tab-paste');
const resumeDropZone = document.getElementById('resume-drop-zone');
const resumeFileInput = document.getElementById('resume-file-input');
const resumeFileInfo = document.getElementById('resume-file-info');
const resumeFileName = document.getElementById('resume-file-name');
const btnClearResumeFile = document.getElementById('btn-clear-resume-file');
const resumeTextInput = document.getElementById('resume-text-input');
const resumePosition = document.getElementById('resume-position');
const resumeQuestion = document.getElementById('resume-question');
const btnStartAnalysis = document.getElementById('btn-start-analysis');
const btnVoice = document.getElementById('btn-voice');
const interviewModal = document.getElementById('interview-modal');
const btnCloseInterview = document.getElementById('btn-close-interview');
const interviewFocus = document.getElementById('interview-focus');
const interviewCount = document.getElementById('interview-count');
const btnStartInterview = document.getElementById('btn-start-interview');

// JD 分析相关 DOM
const btnJd = document.getElementById('btn-jd');
const jdModal = document.getElementById('jd-modal');
const btnCloseJd = document.getElementById('btn-close-jd');
const jdTabs = document.querySelectorAll('.jd-tab');
const jdTabUpload = document.getElementById('jd-tab-upload');
const jdTabPaste = document.getElementById('jd-tab-paste');
const jdDropZone = document.getElementById('jd-drop-zone');
const jdFileInput = document.getElementById('jd-file-input');
const jdFileInfo = document.getElementById('jd-file-info');
const jdFileName = document.getElementById('jd-file-name');
const btnClearJdFile = document.getElementById('btn-clear-jd-file');
const jdTextInput = document.getElementById('jd-text-input');
const jdQuestion = document.getElementById('jd-question');
const btnStartJdAnalysis = document.getElementById('btn-start-jd-analysis');

let pendingImageBase64 = null;
let pendingResumeFile = null;
let pendingJdFile = null;
let isProcessing = false;
let speechRecognition = null;
let isVoiceListening = false;
let isInInterviewMode = false;  // 面试模式标记

// --- 工具函数 ---
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setStatus(text, className) {
    statusIndicator.textContent = `● ${text}`;
    statusIndicator.className = `status-${className}`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function parseIsoDate(value) {
    if (!value) return null;
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
}

function formatSessionBucket(updatedAt) {
    const date = parseIsoDate(updatedAt);
    if (!date) return '更早';

    const now = new Date();
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const startOfYesterday = new Date(startOfToday);
    startOfYesterday.setDate(startOfYesterday.getDate() - 1);
    const startOfLast7Days = new Date(startOfToday);
    startOfLast7Days.setDate(startOfLast7Days.getDate() - 7);

    if (date >= startOfToday) return '今天';
    if (date >= startOfYesterday) return '昨天';
    if (date >= startOfLast7Days) return '最近 7 天';
    return '更早';
}

function formatSessionTime(updatedAt) {
    const date = parseIsoDate(updatedAt);
    if (!date) return '';
    return new Intl.DateTimeFormat('zh-CN', {
        month: 'numeric',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    }).format(date);
}

function getSessionTitle(session) {
    const title = (session.title || '').trim();
    if (title) return title;
    const preview = (session.last_message_preview || '').trim();
    return preview || '新建会话';
}

// --- 简单 Markdown 渲染 ---
function renderMarkdown(text) {
    let html = escapeHtml(text);

    // 安全执行正则替换，避免 LLM 生成的特殊字符导致浏览器 DOMException
    function safeReplace(str, regex, replacer) {
        try {
            return str.replace(regex, replacer);
        } catch (e) {
            console.warn('renderMarkdown 正则替换失败:', e, regex);
            return str; // 跳过该规则，保留原始文本
        }
    }

    // 代码块（必须先处理，避免内部被其他规则干扰）
    html = safeReplace(html, /```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // 行内代码
    html = safeReplace(html, /`([^`]+)`/g, '<code>$1</code>');

    // 链接
    html = safeReplace(html, /\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

    // 表格
    html = safeReplace(html, /^(\|.+\|)\n(\|[-| :]+\|)\n((?:\|.+\|\n?)+)/gm, (match, header, sep, body) => {
        const headers = header.split('|').filter(c => c.trim()).map(c => `<th>${c.trim()}</th>`).join('');
        const rows = body.trim().split('\n').map(row => {
            const cells = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('');
            return `<tr>${cells}</tr>`;
        }).join('');
        return `<table><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
    });

    // 标题
    html = safeReplace(html, /^#### (.+)$/gm, '<h5>$1</h5>');
    html = safeReplace(html, /^### (.+)$/gm, '<h4>$1</h4>');
    html = safeReplace(html, /^## (.+)$/gm, '<h3>$1</h3>');
    html = safeReplace(html, /^# (.+)$/gm, '<h2>$1</h2>');

    // 无序列表
    html = safeReplace(html, /^- (.+)$/gm, '<li>$1</li>');
    // 有序列表
    html = safeReplace(html, /^(\d+)\. (.+)$/gm, '<li>$2</li>');
    // 包裹连续的 li 为 ul
    html = safeReplace(html, /(<li>.*<\/li>\n?)+/g, (match) => `<ul>${match}</ul>`);

    // 粗体 / 斜体
    html = safeReplace(html, /\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = safeReplace(html, /\*(.+?)\*/g, '<em>$1</em>');

    // 水平线
    html = safeReplace(html, /^---$/gm, '<hr>');

    // 段落（处理连续换行）
    html = safeReplace(html, /\n\n+/g, '</p><p>');
    html = safeReplace(html, /\n/g, '<br>');

    return `<p>${html}</p>`;
}

// --- 消息渲染 ---
function appendMessage(role, content, sources = null, routeType = null, isMarkdown = false) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}`;

    let html = '';
    if (role === 'user') {
        html += `<div class="message-content">${escapeHtml(content)}</div>`;
    } else {
        html += `<div class="message-content">`;
        // 路由来源标签
        if (routeType) {
            const routeLabels = {
                kb: '📚 知识库',
                web: '🌐 网络',
                direct: '💬 直接回答',
                resume_analysis: '📋 简历分析',
                jd_analysis: '🎯 岗位分析',
                interview_simulation: '🎤 模拟面试',
                interview_followup: '🎤 面试追问'
            };
            html += `<span class="route-tag ${routeType}">${routeLabels[routeType] || routeType}</span><br>`;
        }
        if (isMarkdown) {
            html += renderMarkdown(content);
        } else {
            html += escapeHtml(content);
        }
        if (sources && sources.length > 0) {
            html += buildSourcesHtml(sources);
        }
        html += `</div>`;
    }
    msgDiv.innerHTML = html;
    chatMessages.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
}

function appendTypingIndicator() {
    const msgDiv = document.createElement('div');
    msgDiv.className = 'message ai';
    msgDiv.id = 'typing-indicator';
    msgDiv.innerHTML = `<div class="message-content"><div class="typing-indicator"><span></span><span></span><span></span></div></div>`;
    chatMessages.appendChild(msgDiv);
    scrollToBottom();
    return msgDiv;
}

function removeTypingIndicator() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

function buildSourcesHtml(sources) {
    let html = '<details class="sources"><summary>📎 引用来源 (' + sources.length + ')</summary><ul>';
    sources.forEach((s, i) => {
        html += `<li><strong>[${i + 1}]</strong> ${escapeHtml(s.source || '未知')} <span style="color:#999">score: ${(s.score || 0).toFixed(3)}</span>`;
        html += `<div class="source-chunk">${escapeHtml(s.content || '')}</div></li>`;
    });
    html += '</ul></details>';
    return html;
}

function buildAgentTimelineHtml(agentSteps) {
    if (!agentSteps || agentSteps.length === 0) return '';

    const labels = {
        qa_flow: 'QA 专家',
        jd_expert: 'JD 专家',
        resume_expert: '简历专家',
        interview_expert: '面试专家',
        react_fallback: 'ReAct 兜底'
    };

    const items = agentSteps.map((step) => {
        const label = labels[step.agent] || step.agent || '处理中';
        const text = step.status === 'done' ? `已完成 ${label}` : `正在调用 ${label}`;
        return `<span class="agent-step ${step.status}">${escapeHtml(text)}</span>`;
    }).join('');

    return `<div class="agent-timeline">${items}</div>`;
}

function buildPlanningMetaHtml(planningMeta) {
    if (!planningMeta || (!planningMeta.task && !planningMeta.questionSignature && !planningMeta.responseMode)) {
        return '';
    }

    const taskLabels = {
        qa: '普通问答',
        resume_analysis: '简历分析',
        jd_analysis: '岗位分析',
        jd_followup: 'JD 追问',
        resume_followup: '简历追问',
        match_followup: '匹配追问',
        interview_simulation: '模拟面试',
        interview_followup: '面试追问',
        react_fallback: 'ReAct 兜底'
    };

    let html = '<div class="planning-meta">';
    if (planningMeta.task) {
        html += `<span class="planning-badge">${escapeHtml(taskLabels[planningMeta.task] || planningMeta.task)}</span>`;
    }
    if (planningMeta.questionSignature) {
        html += `<span class="planning-badge subtle">${escapeHtml(planningMeta.questionSignature)}</span>`;
    }
    if (planningMeta.responseMode) {
        html += `<span class="planning-badge subtle">${escapeHtml(planningMeta.responseMode)}</span>`;
    }
    html += '</div>';
    return html;
}

function getCacheHitStatusText(payload, planningMeta) {
    const task = payload.task || planningMeta?.task || '';
    const mode = payload.response_mode || planningMeta?.responseMode || '';
    const signature = payload.question_signature || planningMeta?.questionSignature || '';

    if (task === 'match_followup') {
        return '命中匹配分析缓存';
    }
    if (task === 'resume_followup') {
        return '命中简历追问缓存';
    }
    if (task === 'jd_followup') {
        return '命中岗位追问缓存';
    }
    if (payload.agent === 'resume_expert') {
        return mode === 'match_brief' || signature.startsWith('match_followup:')
            ? '复用简历与岗位上下文'
            : '复用已有简历分析结果';
    }
    if (payload.agent === 'jd_expert') {
        return '复用已有岗位分析结果';
    }
    return '复用已有结果';
}

function buildCacheHintText(payload) {
    const parts = [];
    if (payload.backend) {
        parts.push(`backend: ${payload.backend}`);
    }
    if (payload.hit_count) {
        parts.push(`hits: ${payload.hit_count}`);
    }
    return parts.join(' · ');
}

// 可观测性标签构建
function buildObsTagsHtml(obsMeta) {
    if (!obsMeta || obsMeta.length === 0) return '';
    const icons = {
        cache_hit: '💾',
        reuse_jd: '🔗',
        reuse_resume: '📎',
        followup_brief: '⚡',
        dedup_extract: '⏭️',
        kb_low_relevance: '🔄',
        tool_running: '🛠️',
        tool_used: '✅',
        interview_start: '🎤',
        interview_progress: '📝',
        interview_score: '📊',
        interview_summary: '🏁',
    };
    const labels = {
        cache_hit: '命中缓存',
        reuse_jd: '复用 JD 上下文',
        reuse_resume: '复用简历结构化',
        followup_brief: 'Follow-up 短答',
        dedup_extract: '跳过重复提取',
        kb_low_relevance: 'KB 降级搜索',
        tool_running: '工具运行中',
        tool_used: '已调用工具',
        interview_start: '模拟面试',
        interview_progress: '面试进度',
        interview_score: '面试评分',
        interview_summary: '面试复盘',
    };
    const items = obsMeta.map(tag => {
        const icon = icons[tag.type] || 'ℹ️';
        const label = labels[tag.type] || tag.type;
        let detail = '';
        if (tag.detail) detail = ` <span class="obs-detail">${escapeHtml(tag.detail)}</span>`;
        return `<span class="obs-tag ${tag.type}">${icon} ${label}${detail}</span>`;
    }).join('');
    return `<div class="obs-tags">${items}</div>`;
}

function renderAiMessageContent({ routeType, fullAnswer, currentSources, agentSteps, planningMeta, obsMeta }) {
    const routeLabels = {
        kb: '📚 知识库',
        web: '🌐 网络',
        direct: '💬 直接回答',
        resume_analysis: '📋 简历分析',
        jd_analysis: '🎯 岗位分析',
        jd_followup: '🎯 JD 追问',
        resume_followup: '📋 简历追问',
        match_followup: '🧩 匹配追问',
        interview_simulation: '🎤 模拟面试',
        interview_followup: '🎤 面试追问',
        react_fallback: '🛠️ ReAct 兜底'
    };

    const routeHtml = routeType ? `<span class="route-tag ${routeType}">${routeLabels[routeType] || routeType}</span><br>` : '';
    const planningHtml = buildPlanningMetaHtml(planningMeta);
    const obsHtml = buildObsTagsHtml(obsMeta);
    const timelineHtml = buildAgentTimelineHtml(agentSteps);
    const sourcesHtml = currentSources.length > 0 ? buildSourcesHtml(currentSources) : '';

    if (!fullAnswer) {
        return routeHtml + planningHtml + obsHtml + timelineHtml + sourcesHtml;
    }

    try {
        return routeHtml + planningHtml + obsHtml + timelineHtml + renderMarkdown(fullAnswer) + sourcesHtml;
    } catch (renderErr) {
        console.warn('Markdown 渲染异常，降级为纯文本:', renderErr);
        return routeHtml + planningHtml + obsHtml + timelineHtml + '<p>' + escapeHtml(fullAnswer) + '</p>' + sourcesHtml;
    }
}

// --- Agent 多轮对话（流式） ---
async function sendChat(question) {
    if (isProcessing || !question.trim()) return;
    isProcessing = true;
    btnSend.disabled = true;
    setStatus('思考中', 'processing');

    appendMessage('user', question);
    appendTypingIndicator();

    try {
        const res = await fetch(`${API_BASE}/agent/chat/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question,
                session_id: sessionId
            })
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            removeTypingIndicator();
            appendMessage('ai', `❌ 请求失败：${errData.detail || res.statusText}`);
            setStatus('错误', 'error');
            return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullAnswer = '';
        let currentSources = [];
        let currentRoute = 'direct';
        let agentSteps = [];
        let planningMeta = { task: '', questionSignature: '', responseMode: '' };
        let obsMeta = [];  // 可观测性标签
        let msgDiv = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            // 解析 SSE 数据（可能跨多个 chunk，需要缓冲处理）
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';  // 保留最后一个不完整的行

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;

                try {
                    const payload = JSON.parse(line.slice(6));

                    if (payload.type === 'route') {
                        // 路由决策完成，移除 typing indicator，创建 AI 消息容器
                        removeTypingIndicator();
                        currentRoute = payload.route;
                        msgDiv = appendMessage('ai', '', [], currentRoute, true);
                        const contentEl = msgDiv.querySelector('.message-content');
                        contentEl.innerHTML = renderAiMessageContent({
                            routeType: currentRoute,
                            fullAnswer,
                            currentSources,
                            agentSteps,
                            planningMeta,
                            obsMeta,
                        });
                        setStatus('生成中', 'processing');

                    } else if (payload.type === 'planning') {
                        planningMeta = {
                            task: payload.task || planningMeta.task,
                            questionSignature: payload.question_signature || planningMeta.questionSignature,
                            responseMode: payload.response_mode || planningMeta.responseMode,
                        };
                        // follow-up 短答模式标签
                        if (payload.response_mode && ['followup_brief', 'match_brief'].includes(payload.response_mode)) {
                            if (!obsMeta.find(t => t.type === 'followup_brief')) {
                                obsMeta.push({ type: 'followup_brief', detail: payload.response_mode });
                            }
                        }
                        if (payload.task) {
                            currentRoute = payload.task;
                        }
                        if (!msgDiv) {
                            removeTypingIndicator();
                            msgDiv = appendMessage('ai', '', [], currentRoute, true);
                        }
                        const contentEl = msgDiv.querySelector('.message-content');
                        contentEl.innerHTML = renderAiMessageContent({
                            routeType: currentRoute,
                            fullAnswer,
                            currentSources,
                            agentSteps,
                            planningMeta,
                            obsMeta,
                        });
                        setStatus('规划中', 'processing');

                    } else if (payload.type === 'agent_start') {
                        agentSteps = agentSteps.filter(step => !(step.agent === payload.agent && step.status === 'running'));
                        agentSteps.push({ agent: payload.agent, status: 'running' });
                        const statusLabels = {
                            qa_flow: '正在检索知识库',
                            jd_expert: '正在分析岗位描述',
                            resume_expert: '正在评估简历',
                            interview_expert: '正在进行模拟面试',
                            react_fallback: '正在组合工具处理非标准请求'
                        };
                        setStatus(statusLabels[payload.agent] || '处理中', 'processing');
                        if (msgDiv) {
                            const contentEl = msgDiv.querySelector('.message-content');
                            contentEl.innerHTML = renderAiMessageContent({
                                routeType: currentRoute,
                                fullAnswer,
                                currentSources,
                                agentSteps,
                                planningMeta,
                                obsMeta,
                            });
                        }

                    } else if (payload.type === 'agent_result') {
                        agentSteps = agentSteps.map(step =>
                            step.agent === payload.agent ? { ...step, status: 'done' } : step
                        );
                        if (msgDiv) {
                            const contentEl = msgDiv.querySelector('.message-content');
                            contentEl.innerHTML = renderAiMessageContent({
                                routeType: currentRoute,
                                fullAnswer,
                                currentSources,
                                agentSteps,
                                planningMeta,
                                obsMeta,
                            });
                        }

                    } else if (payload.type === 'tool_start') {
                        setStatus(`正在调用工具：${payload.tool || 'unknown'}`, 'processing');
                        obsMeta = obsMeta.filter(t => !(t.type === 'tool_running' && t.detail === payload.tool));
                        obsMeta.push({ type: 'tool_running', detail: payload.tool || 'unknown' });
                        if (msgDiv) {
                            const contentEl = msgDiv.querySelector('.message-content');
                            contentEl.innerHTML = renderAiMessageContent({
                                routeType: currentRoute,
                                fullAnswer,
                                currentSources,
                                agentSteps,
                                planningMeta,
                                obsMeta,
                            });
                        }

                    } else if (payload.type === 'tool_result') {
                        setStatus(`已完成工具：${payload.tool || 'unknown'}`, 'processing');
                        obsMeta = obsMeta.filter(t => !(t.type === 'tool_running' && t.detail === payload.tool));
                        if (!obsMeta.find(t => t.type === 'tool_used' && t.detail === payload.tool)) {
                            obsMeta.push({ type: 'tool_used', detail: payload.tool || 'unknown' });
                        }
                        if (msgDiv) {
                            const contentEl = msgDiv.querySelector('.message-content');
                            contentEl.innerHTML = renderAiMessageContent({
                                routeType: currentRoute,
                                fullAnswer,
                                currentSources,
                                agentSteps,
                                planningMeta,
                                obsMeta,
                            });
                        }

                    } else if (payload.type === 'tool_cache_hit') {
                        setStatus(`命中工具缓存：${payload.tool || 'unknown'}`, 'processing');
                        if (!obsMeta.find(t => t.type === 'cache_hit' && t.detail === `tool:${payload.tool || 'unknown'}`)) {
                            obsMeta.push({ type: 'cache_hit', detail: `tool:${payload.tool || 'unknown'}` });
                        }
                        if (msgDiv) {
                            const contentEl = msgDiv.querySelector('.message-content');
                            contentEl.innerHTML = renderAiMessageContent({
                                routeType: currentRoute,
                                fullAnswer,
                                currentSources,
                                agentSteps,
                                planningMeta,
                                obsMeta,
                            });
                        }

                    } else if (payload.type === 'agent_cache_hit') {
                        // 可观测性标签
                        if (!obsMeta.find(t => t.type === 'cache_hit')) {
                            const detail = payload.backend
                                ? `${payload.backend}${payload.hit_count ? ' · hits:' + payload.hit_count : ''}`
                                : '';
                            obsMeta.push({ type: 'cache_hit', detail });
                        }
                        setStatus(getCacheHitStatusText(payload, planningMeta), 'processing');
                        if (msgDiv) {
                            const hint = buildCacheHintText(payload);
                            const contentEl = msgDiv.querySelector('.message-content');
                            contentEl.innerHTML = renderAiMessageContent({
                                routeType: currentRoute,
                                fullAnswer,
                                currentSources,
                                agentSteps,
                                planningMeta,
                                obsMeta,
                            });
                        }

                    } else if (payload.type === 'status') {
                        if (payload.content) {
                            setStatus(payload.content, 'processing');
                            // 从 status 文本推断可观测性标签
                            const content = payload.content;
                            if (content.includes('复用') && content.includes('JD') && !obsMeta.find(t => t.type === 'reuse_jd')) {
                                obsMeta.push({ type: 'reuse_jd' });
                            }
                            if (content.includes('复用') && content.includes('简历') && !obsMeta.find(t => t.type === 'reuse_resume')) {
                                obsMeta.push({ type: 'reuse_resume' });
                            }
                            if (content.includes('跳过') && content.includes('提取') && !obsMeta.find(t => t.type === 'dedup_extract')) {
                                obsMeta.push({ type: 'dedup_extract' });
                            }
                            if (content.includes('降级') && !obsMeta.find(t => t.type === 'kb_low_relevance')) {
                                obsMeta.push({ type: 'kb_low_relevance' });
                            }
                        }

                    } else if (payload.type === 'interview_progress') {
                        // 面试进度事件
                        const phase = payload.phase || '';
                        const qIdx = payload.question_index || 0;
                        const total = payload.total_questions || 0;
                        const curScore = payload.current_score || 0;
                        const avgScore = payload.average_score || 0;
                        if (phase === 'start') {
                            if (!obsMeta.find(t => t.type === 'interview_start')) {
                                obsMeta.push({ type: 'interview_start', detail: total ? `${total} 题` : '' });
                            }
                        } else if (phase === 'question') {
                            if (!obsMeta.find(t => t.type === 'interview_progress')) {
                                obsMeta.push({ type: 'interview_progress', detail: `第 ${qIdx + 1}/${total} 题` });
                            } else {
                                const tag = obsMeta.find(t => t.type === 'interview_progress');
                                tag.detail = `第 ${qIdx + 1}/${total} 题`;
                            }
                        } else if (phase === 'evaluated') {
                            if (!obsMeta.find(t => t.type === 'interview_score')) {
                                obsMeta.push({ type: 'interview_score', detail: `本轮 ${curScore} · 均分 ${avgScore}` });
                            } else {
                                const tag = obsMeta.find(t => t.type === 'interview_score');
                                tag.detail = `本轮 ${curScore} · 均分 ${avgScore}`;
                            }
                            if (obsMeta.find(t => t.type === 'interview_progress')) {
                                const tag = obsMeta.find(t => t.type === 'interview_progress');
                                tag.detail = `第 ${qIdx + 1}/${total} 题`;
                            }
                            // 面试模式下，评分后自动 focus 输入框
                            if (isInInterviewMode) {
                                messageInput.focus();
                            }
                        } else if (phase === 'summary') {
                            if (!obsMeta.find(t => t.type === 'interview_summary')) {
                                obsMeta.push({ type: 'interview_summary', detail: '生成复盘报告' });
                            }
                        }
                        // 实时更新消息区域的标签
                        if (msgDiv) {
                            const contentEl = msgDiv.querySelector('.message-content');
                            contentEl.innerHTML = renderAiMessageContent({
                                routeType: currentRoute,
                                fullAnswer,
                                currentSources,
                                agentSteps,
                                planningMeta,
                                obsMeta,
                            });
                        }

                    } else if (payload.type === 'sources') {
                        // 检索来源
                        currentSources = payload.sources || [];
                        // 更新消息气泡的 sources
                        if (msgDiv) {
                            const contentEl = msgDiv.querySelector('.message-content');
                            contentEl.innerHTML = renderAiMessageContent({
                                routeType: currentRoute,
                                fullAnswer,
                                currentSources,
                                agentSteps,
                                planningMeta,
                                obsMeta,
                            });
                        }

                    } else if (payload.type === 'token') {
                        // 增量文本
                        fullAnswer += payload.content;
                        if (msgDiv) {
                            const contentEl = msgDiv.querySelector('.message-content');
                            contentEl.innerHTML = renderAiMessageContent({
                                routeType: currentRoute,
                                fullAnswer,
                                currentSources,
                                agentSteps,
                                planningMeta,
                                obsMeta,
                            });
                            scrollToBottom();
                        }

                    } else if (payload.type === 'done') {
                        // 完成
                        if (!fullAnswer && payload.answer && msgDiv) {
                            fullAnswer = payload.answer;
                            const contentEl = msgDiv.querySelector('.message-content');
                            contentEl.innerHTML = renderAiMessageContent({
                                routeType: currentRoute,
                                fullAnswer,
                                currentSources,
                                agentSteps,
                                planningMeta,
                                obsMeta,
                            });
                        }
                        if (payload.session_id && payload.session_id !== sessionId) {
                            sessionId = payload.session_id;
                            localStorage.setItem('agent_session_id', sessionId);
                        }
                        // 检测面试结束 → 退出面试模式
                        if (isInInterviewMode && fullAnswer && (fullAnswer.includes('模拟面试结束') || fullAnswer.includes('模拟面试已结束'))) {
                            isInInterviewMode = false;
                            messageInput.placeholder = '输入你的问题…';
                        }
                        // 面试模式下评分后自动 focus
                        if (isInInterviewMode) {
                            messageInput.focus();
                        }
                        setStatus('就绪', 'ready');
                        // 对话完成后刷新会话列表
                        loadSessionList();

                    } else if (payload.type === 'error') {
                        removeTypingIndicator();
                        appendMessage('ai', `❌ 服务错误：${payload.message}`);
                        setStatus('错误', 'error');
                    }
                } catch (parseErr) {
                    console.warn('SSE 解析失败:', parseErr, line);
                }
            }
        }

        setStatus('就绪', 'ready');
    } catch (err) {
        removeTypingIndicator();
        appendMessage('ai', `❌ 网络错误：${err.message}`);
        setStatus('错误', 'error');
    } finally {
        isProcessing = false;
        btnSend.disabled = false;
    }
}

// --- 图片即时问答 ---
async function sendImageChat(question, imageBase64) {
    if (isProcessing) return;
    isProcessing = true;
    btnSend.disabled = true;
    setStatus('视觉理解中', 'processing');

    const msgDiv = document.createElement('div');
    msgDiv.className = 'message user';
    msgDiv.innerHTML = `<div class="message-content">${escapeHtml(question || '请描述这张图片')}<img class="message-image" src="${imageBase64}" alt="用户图片"></div>`;
    chatMessages.appendChild(msgDiv);
    scrollToBottom();

    appendTypingIndicator();

    try {
        const res = await fetch(`${API_BASE}/chat/image`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question || '请描述这张图片的内容',
                image_base64: imageBase64.split(',')[1]
            })
        });
        const data = await res.json();
        removeTypingIndicator();

        if (res.ok) {
            appendMessage('ai', data.answer, null, null, true);
        } else {
            appendMessage('ai', `❌ 请求失败：${data.detail || '未知错误'}`);
            setStatus('错误', 'error');
            return;
        }
        setStatus('就绪', 'ready');
    } catch (err) {
        removeTypingIndicator();
        appendMessage('ai', `❌ 网络错误：${err.message}`);
        setStatus('错误', 'error');
    } finally {
        isProcessing = false;
        btnSend.disabled = false;
    }
}

// --- 简历分析（流式 SSE） ---
async function analyzeResume(file, text) {
    if (isProcessing) return;
    isProcessing = true;
    btnStartAnalysis.disabled = true;
    btnStartAnalysis.textContent = '⏳ 分析中...';

    const question = resumeQuestion.value.trim() || '请对我的简历进行全面分析评估';
    const targetPosition = resumePosition.value.trim();

    // 显示用户消息
    let userMsg = '📋 请分析我的简历';
    if (file) userMsg += `（文件：${file.name}）`;
    if (targetPosition) userMsg += `，目标岗位：${targetPosition}`;
    if (question !== '请对我的简历进行全面分析评估') userMsg += `\n要求：${question}`;

    appendMessage('user', userMsg);
    appendTypingIndicator();
    resumeModal.classList.add('hidden');

    try {
        let res;
        if (file) {
            // 文件上传方式
            const formData = new FormData();
            formData.append('file', file);
            formData.append('question', question);
            formData.append('session_id', sessionId);
            if (targetPosition) formData.append('target_position', targetPosition);

            res = await fetch(`${API_BASE}/agent/resume-upload`, {
                method: 'POST',
                body: formData
            });
        } else {
            // 文本粘贴方式
            res = await fetch(`${API_BASE}/agent/resume-analysis`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    resume_text: text,
                    question,
                    session_id: sessionId,
                    target_position: targetPosition
                })
            });
        }

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            removeTypingIndicator();
            appendMessage('ai', `❌ 简历分析失败：${errData.detail || res.statusText}`);
            setStatus('错误', 'error');
            return;
        }

        // SSE 流式读取
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullAnswer = '';
        let currentSources = [];
        let msgDiv = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;

                try {
                    const payload = JSON.parse(line.slice(6));

                    if (payload.type === 'extracted') {
                        // 简历提取完成
                        removeTypingIndicator();
                        msgDiv = appendMessage('ai', '', [], 'resume_analysis', true);
                        setStatus('生成中', 'processing');

                    } else if (payload.type === 'sources') {
                        // JD 来源
                        currentSources = payload.sources || [];
                        if (msgDiv) {
                            const sourcesHtml = buildSourcesHtml(currentSources);
                            const existingContent = msgDiv.querySelector('.message-content').innerHTML;
                            msgDiv.querySelector('.message-content').innerHTML = existingContent + sourcesHtml;
                        }

                    } else if (payload.type === 'status') {
                        if (payload.content) {
                            setStatus(payload.content, 'processing');
                        }

                    } else if (payload.type === 'token') {
                        // 增量文本
                        fullAnswer += payload.content;
                        if (msgDiv) {
                            const contentEl = msgDiv.querySelector('.message-content');
                            const sourcesHtml = currentSources.length > 0 ? buildSourcesHtml(currentSources) : '';
                            try {
                                contentEl.innerHTML = renderMarkdown(fullAnswer) + sourcesHtml;
                            } catch (renderErr) {
                                console.warn('简历分析 Markdown 渲染异常，降级为纯文本:', renderErr);
                                contentEl.innerHTML = '<p>' + escapeHtml(fullAnswer) + '</p>' + sourcesHtml;
                            }
                            scrollToBottom();
                        }

                    } else if (payload.type === 'done') {
                        // 完成
                        if (payload.session_id && payload.session_id !== sessionId) {
                            sessionId = payload.session_id;
                            localStorage.setItem('agent_session_id', sessionId);
                        }
                        // 如果流式过程中没有 token，用 done 中的 answer
                        if (!fullAnswer && payload.answer) {
                            fullAnswer = payload.answer;
                            if (msgDiv) {
                                const sourcesHtml = currentSources.length > 0 ? buildSourcesHtml(currentSources) : '';
                                try {
                                    msgDiv.querySelector('.message-content').innerHTML = renderMarkdown(fullAnswer) + sourcesHtml;
                                } catch (renderErr) {
                                    console.warn('简历分析 done Markdown 渲染异常，降级为纯文本:', renderErr);
                                    msgDiv.querySelector('.message-content').innerHTML = '<p>' + escapeHtml(fullAnswer) + '</p>' + sourcesHtml;
                                }
                            }
                        }
                        setStatus('就绪', 'ready');
                        // 分析完成后刷新会话列表
                        loadSessionList();

                    } else if (payload.type === 'error') {
                        removeTypingIndicator();
                        appendMessage('ai', `❌ ${payload.message}`);
                        setStatus('错误', 'error');
                    }
                } catch (parseErr) {
                    console.warn('简历分析 SSE 解析失败:', parseErr, line);
                }
            }
        }

        // 如果全程没有收到 extracted 事件（说明完全没数据），确保移除 typing
        removeTypingIndicator();
        if (!fullAnswer) {
            appendMessage('ai', '❌ 简历分析未返回结果');
            setStatus('错误', 'error');
        }
    } catch (err) {
        removeTypingIndicator();
        appendMessage('ai', `❌ 网络错误：${err.message}`);
        setStatus('错误', 'error');
    } finally {
        isProcessing = false;
        btnStartAnalysis.disabled = false;
        btnStartAnalysis.textContent = '🔍 开始分析';
    }
}

// --- 文件上传（知识库） ---
async function uploadFiles(files) {
    if (isProcessing || files.length === 0) return;
    isProcessing = true;

    uploadModal.classList.remove('hidden');
    btnCloseModal.classList.add('hidden');
    progressFill.style.width = '0%';

    // FileList 转数组，防止索引访问异常
    const fileArray = Array.from(files);
    const total = fileArray.length;
    let successCount = 0;

    for (let i = 0; i < total; i++) {
        const file = fileArray[i];
        uploadStatusText.textContent = `正在处理：${file.name} (${i + 1}/${total})`;
        progressFill.style.width = `${((i) / total) * 100}%`;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch(`${API_BASE}/ingest/file`, {
                method: 'POST',
                body: formData
            });
            const data = await res.json();

            if (res.ok) {
                successCount++;
            }
        } catch (err) {
            // 静默处理
        }
    }

    progressFill.style.width = '100%';
    uploadStatusText.textContent = `处理完成：${successCount}/${total} 成功`;
    btnCloseModal.classList.remove('hidden');
    setStatus('就绪', 'ready');
    isProcessing = false;
    console.log(`[Upload] 全部完成: ${successCount}/${total}`);
}

// --- 会话列表 ---
async function loadSessionList() {
    try {
        const res = await fetch(`${API_BASE}/agent/sessions?limit=50`);
        if (!res.ok) return;
        const data = await res.json();

        sessionList.innerHTML = '';

        if (!data || data.length === 0) {
            sessionList.innerHTML = '<p class="empty-hint">暂无历史会话</p>';
            return;
        }

        const grouped = new Map();
        data.forEach(session => {
            const bucket = formatSessionBucket(session.updated_at || session.created_at);
            if (!grouped.has(bucket)) grouped.set(bucket, []);
            grouped.get(bucket).push(session);
        });

        const bucketOrder = ['今天', '昨天', '最近 7 天', '更早'];
        bucketOrder.forEach(bucket => {
            const sessions = grouped.get(bucket);
            if (!sessions || sessions.length === 0) return;

            const section = document.createElement('section');
            section.className = 'session-group';

            const heading = document.createElement('div');
            heading.className = 'session-group-title';
            heading.textContent = bucket;
            section.appendChild(heading);

            sessions.forEach(session => {
                const div = document.createElement('div');
                div.className = `session-item${session.session_id === sessionId ? ' active' : ''}`;
                div.dataset.sessionId = session.session_id;

                const title = getSessionTitle(session);
                const preview = session.last_message_preview || '';
                const metaParts = [];
                if (session.updated_at || session.created_at) {
                    metaParts.push(formatSessionTime(session.updated_at || session.created_at));
                }
                if (session.message_count) {
                    metaParts.push(`${session.message_count}条消息`);
                }

                const dataBadges = [];
                if (session.has_resume_data) dataBadges.push('<span class="session-badge resume">简历</span>');
                if (session.has_jd_data) dataBadges.push('<span class="session-badge jd">JD</span>');

                div.innerHTML = `
                    <div class="session-main">
                        <div class="session-title-row">
                            <div class="session-title">${escapeHtml(title)}</div>
                            <button class="session-more" type="button" tabindex="-1" aria-label="会话操作">⋯</button>
                        </div>
                        ${preview && preview !== title ? `<div class="session-preview">${escapeHtml(preview)}</div>` : ''}
                        <div class="session-meta">
                            ${metaParts.length > 0 ? `<span class="session-msg-count">${escapeHtml(metaParts.join(' · '))}</span>` : ''}
                            ${dataBadges.join('')}
                        </div>
                    </div>
                `;

                const moreBtn = div.querySelector('.session-more');
                if (moreBtn) {
                    moreBtn.addEventListener('click', (event) => {
                        event.stopPropagation();
                    });
                }
                div.addEventListener('click', () => switchToSession(session.session_id));
                section.appendChild(div);
            });

            sessionList.appendChild(section);
        });
    } catch (e) {
        console.warn('加载会话列表失败:', e);
    }
}

async function switchToSession(targetSessionId) {
    if (isProcessing || targetSessionId === sessionId) return;
    isProcessing = true;

    try {
        // 获取目标会话的消息历史
        const res = await fetch(`${API_BASE}/agent/sessions/${targetSessionId}/messages`);
        if (!res.ok) {
            console.warn('获取会话消息失败:', res.status);
            isProcessing = false;
            return;
        }
        const data = await res.json();

        // 切换 sessionId
        sessionId = targetSessionId;
        localStorage.setItem('agent_session_id', sessionId);

        // 清空当前聊天区域
        chatMessages.innerHTML = '';

        // 渲染历史消息
        if (data.messages && data.messages.length > 0) {
            for (const msg of data.messages) {
                if (msg.role === 'user') {
                    appendMessage('user', msg.content);
                } else if (msg.role === 'assistant') {
                    const routeType = msg.route_type || msg.task_type || null;
                    appendMessage('ai', msg.content, [], routeType, true);
                }
            }
        } else {
            showWelcome();
        }

        // 更新侧边栏选中状态
        sessionList.querySelectorAll('.session-item').forEach(el => {
            el.classList.toggle('active', el.dataset.sessionId === targetSessionId);
        });

        setStatus('就绪', 'ready');
    } catch (err) {
        console.error('切换会话失败:', err);
    } finally {
        isProcessing = false;
    }
}

// --- 图片预览管理 ---
function handleImageSelect(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        pendingImageBase64 = e.target.result;
        previewImg.src = pendingImageBase64;
        imagePreview.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

// --- 简历分析模态框逻辑 ---

// Tab 切换
resumeTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        resumeTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        const tabName = tab.dataset.tab;
        document.querySelectorAll('.resume-tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(`resume-tab-${tabName}`).classList.add('active');
    });
});

// 打开简历分析模态框
btnResume.addEventListener('click', () => {
    resumeModal.classList.remove('hidden');
});

// 关闭简历分析模态框
btnCloseResume.addEventListener('click', () => {
    resumeModal.classList.add('hidden');
});

// 拖拽上传
resumeDropZone.addEventListener('click', () => resumeFileInput.click());

resumeDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    resumeDropZone.classList.add('drag-over');
});

resumeDropZone.addEventListener('dragleave', () => {
    resumeDropZone.classList.remove('drag-over');
});

resumeDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    resumeDropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleResumeFileSelect(e.dataTransfer.files[0]);
    }
});

resumeFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleResumeFileSelect(e.target.files[0]);
        e.target.value = '';
    }
});

function handleResumeFileSelect(file) {
    const allowed = ['.pdf', '.png', '.jpg', '.jpeg', '.txt', '.md'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowed.includes(ext)) {
        alert('不支持的文件格式，请上传 PDF、图片或文本文件');
        return;
    }
    pendingResumeFile = file;
    resumeFileName.textContent = `📎 ${file.name} (${(file.size / 1024).toFixed(1)}KB)`;
    resumeDropZone.classList.add('hidden');
    resumeFileInfo.classList.remove('hidden');
}

btnClearResumeFile.addEventListener('click', () => {
    pendingResumeFile = null;
    resumeDropZone.classList.remove('hidden');
    resumeFileInfo.classList.add('hidden');
});

// 开始分析
btnStartAnalysis.addEventListener('click', () => {
    const activeTab = document.querySelector('.resume-tab.active').dataset.tab;

    if (activeTab === 'upload') {
        if (!pendingResumeFile) {
            alert('请先上传简历文件');
            return;
        }
        analyzeResume(pendingResumeFile, null);
    } else {
        const text = resumeTextInput.value.trim();
        if (!text) {
            alert('请粘贴简历内容');
            return;
        }
        analyzeResume(null, text);
    }
});

// --- JD 分析模态框逻辑 ---

// Tab 切换
jdTabs.forEach(tab => {
    tab.addEventListener('click', () => {
        jdTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        const tabName = tab.dataset.tab;
        document.querySelectorAll('.jd-tab-content').forEach(c => c.classList.remove('active'));
        document.getElementById(`jd-tab-${tabName}`).classList.add('active');
    });
});

// 打开 JD 分析模态框
btnJd.addEventListener('click', () => {
    jdModal.classList.remove('hidden');
});

// 关闭 JD 分析模态框
btnCloseJd.addEventListener('click', () => {
    jdModal.classList.add('hidden');
});

// 拖拽上传
jdDropZone.addEventListener('click', () => jdFileInput.click());

jdDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    jdDropZone.classList.add('drag-over');
});

jdDropZone.addEventListener('dragleave', () => {
    jdDropZone.classList.remove('drag-over');
});

jdDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    jdDropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleJdFileSelect(e.dataTransfer.files[0]);
    }
});

jdFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleJdFileSelect(e.target.files[0]);
        e.target.value = '';
    }
});

function handleJdFileSelect(file) {
    const allowed = ['.pdf', '.png', '.jpg', '.jpeg', '.txt', '.md'];
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!allowed.includes(ext)) {
        alert('不支持的文件格式，请上传 PDF、图片或文本文件');
        return;
    }
    pendingJdFile = file;
    jdFileName.textContent = `📎 ${file.name} (${(file.size / 1024).toFixed(1)}KB)`;
    jdDropZone.classList.add('hidden');
    jdFileInfo.classList.remove('hidden');
}

btnClearJdFile.addEventListener('click', () => {
    pendingJdFile = null;
    jdDropZone.classList.remove('hidden');
    jdFileInfo.classList.add('hidden');
});

// --- JD 分析（流式 SSE） ---
async function analyzeJD(file, text) {
    if (isProcessing) return;
    isProcessing = true;
    btnStartJdAnalysis.disabled = true;
    btnStartJdAnalysis.textContent = '⏳ 分析中...';

    const question = jdQuestion.value.trim() || '请分析该岗位的核心要求并给出简历写作建议';

    // 显示用户消息
    let userMsg = '🎯 请分析以下岗位JD';
    if (file) userMsg += `（文件：${file.name}）`;
    if (question !== '请分析该岗位的核心要求并给出简历写作建议') userMsg += `\n要求：${question}`;

    appendMessage('user', userMsg);
    appendTypingIndicator();
    jdModal.classList.add('hidden');

    try {
        let res;
        if (file) {
            // 文件上传方式
            const formData = new FormData();
            formData.append('file', file);
            formData.append('question', question);
            formData.append('session_id', sessionId);

            res = await fetch(`${API_BASE}/agent/jd-upload`, {
                method: 'POST',
                body: formData
            });
        } else {
            // 文本粘贴方式
            res = await fetch(`${API_BASE}/agent/jd-analysis`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    jd_text: text,
                    question,
                    session_id: sessionId
                })
            });
        }

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            removeTypingIndicator();
            appendMessage('ai', `❌ JD 分析失败：${errData.detail || res.statusText}`);
            setStatus('错误', 'error');
            return;
        }

        // SSE 流式读取
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullAnswer = '';
        let msgDiv = null;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;

                try {
                    const payload = JSON.parse(line.slice(6));

                    if (payload.type === 'extracted') {
                        // JD 提取完成
                        removeTypingIndicator();
                        msgDiv = appendMessage('ai', '', [], 'jd_analysis', true);
                        setStatus('生成中', 'processing');

                    } else if (payload.type === 'token') {
                        // 增量文本
                        fullAnswer += payload.content;
                        if (msgDiv) {
                            const contentEl = msgDiv.querySelector('.message-content');
                            try {
                                contentEl.innerHTML = renderMarkdown(fullAnswer);
                            } catch (renderErr) {
                                console.warn('JD 分析 Markdown 渲染异常，降级为纯文本:', renderErr);
                                contentEl.innerHTML = '<p>' + escapeHtml(fullAnswer) + '</p>';
                            }
                            scrollToBottom();
                        }

                    } else if (payload.type === 'status') {
                        if (payload.content) {
                            setStatus(payload.content, 'processing');
                        }

                    } else if (payload.type === 'done') {
                        // 完成
                        if (payload.session_id && payload.session_id !== sessionId) {
                            sessionId = payload.session_id;
                            localStorage.setItem('agent_session_id', sessionId);
                        }
                        // 如果流式过程中没有 token，用 done 中的 answer
                        if (!fullAnswer && payload.answer) {
                            fullAnswer = payload.answer;
                            if (msgDiv) {
                                try {
                                    msgDiv.querySelector('.message-content').innerHTML = renderMarkdown(fullAnswer);
                                } catch (renderErr) {
                                    console.warn('JD 分析 done Markdown 渲染异常，降级为纯文本:', renderErr);
                                    msgDiv.querySelector('.message-content').innerHTML = '<p>' + escapeHtml(fullAnswer) + '</p>';
                                }
                            }
                        }
                        setStatus('就绪', 'ready');
                        // JD 分析完成后刷新会话列表
                        loadSessionList();

                    } else if (payload.type === 'error') {
                        removeTypingIndicator();
                        appendMessage('ai', `❌ ${payload.message}`);
                        setStatus('错误', 'error');
                    }
                } catch (parseErr) {
                    console.warn('JD 分析 SSE 解析失败:', parseErr, line);
                }
            }
        }

        // 如果全程没有收到 extracted 事件，确保移除 typing
        removeTypingIndicator();
        if (!fullAnswer) {
            appendMessage('ai', '❌ JD 分析未返回结果');
            setStatus('错误', 'error');
        }
    } catch (err) {
        removeTypingIndicator();
        appendMessage('ai', `❌ 网络错误：${err.message}`);
        setStatus('错误', 'error');
    } finally {
        isProcessing = false;
        btnStartJdAnalysis.disabled = false;
        btnStartJdAnalysis.textContent = '🔍 开始分析';
    }
}

btnStartJdAnalysis.addEventListener('click', () => {
    const activeTab = document.querySelector('.jd-tab.active').dataset.tab;

    if (activeTab === 'upload') {
        if (!pendingJdFile) {
            alert('请先上传 JD 文件');
            return;
        }
        analyzeJD(pendingJdFile, null);
    } else {
        const text = jdTextInput.value.trim();
        if (!text) {
            alert('请粘贴 JD 内容');
            return;
        }
        analyzeJD(null, text);
    }
});

// --- 通用事件绑定 ---
btnSend.addEventListener('click', () => {
    const text = messageInput.value.trim();
    if (pendingImageBase64) {
        sendImageChat(text, pendingImageBase64);
        clearImagePreview();
    } else if (text) {
        sendChat(text);
        messageInput.value = '';
        messageInput.style.height = 'auto';
    }
});

messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        btnSend.click();
    }
});

// 自动增高
messageInput.addEventListener('input', () => {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
});

btnUpload.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        uploadFiles(e.target.files);
        e.target.value = '';
    }
});

btnRefreshSessions.addEventListener('click', () => {
    loadSessionList();
});

btnImage.addEventListener('click', () => imageInput.click());
imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleImageSelect(e.target.files[0]);
        e.target.value = '';
    }
});

function clearImagePreview() {
    pendingImageBase64 = null;
    previewImg.src = '';
    imagePreview.classList.add('hidden');
}

function buildInterviewPrompt() {
    const focus = interviewFocus.value.trim();
    const count = interviewCount.value.trim();
    let prompt = '请开始一轮模拟面试。请结合当前会话中的 JD、简历和历史对话生成面试题，并在我每次回答后给出评分、分析和下一题。';
    if (focus) {
        prompt += ` 面试重点：${focus}。`;
    }
    if (count) {
        prompt += ` 本轮请控制在 ${count} 题左右。`;
    }
    return prompt;
}

function closeInterviewModal() {
    interviewModal.classList.add('hidden');
}

function initVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        btnVoice.disabled = true;
        btnVoice.title = '当前浏览器不支持语音输入（请使用 Chrome 或 Edge）';
        btnVoice.style.opacity = '0.4';
        // 在语音按钮旁显示降级提示
        const hint = document.createElement('span');
        hint.textContent = '请使用 Chrome/Edge';
        hint.style.cssText = 'font-size:11px;color:var(--text-secondary);margin-left:4px;';
        btnVoice.parentNode.insertBefore(hint, btnVoice.nextSibling);
        return;
    }

    speechRecognition = new SpeechRecognition();
    speechRecognition.lang = 'zh-CN';
    speechRecognition.continuous = false;
    speechRecognition.interimResults = true;

    speechRecognition.onstart = () => {
        isVoiceListening = true;
        btnVoice.classList.add('listening');
        setStatus('语音输入中', 'processing');
    };

    speechRecognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }
        messageInput.value = transcript.trim();
        messageInput.dispatchEvent(new Event('input'));
    };

    speechRecognition.onend = () => {
        isVoiceListening = false;
        btnVoice.classList.remove('listening');
        setStatus('就绪', 'ready');
    };

    speechRecognition.onerror = (event) => {
        isVoiceListening = false;
        btnVoice.classList.remove('listening');
        if (event.error === 'not-allowed') {
            setStatus('麦克风权限被拒绝', 'error');
        } else {
            setStatus('语音识别失败', 'error');
        }
    };
}

clearImageBtn.addEventListener('click', clearImagePreview);

btnCloseModal.addEventListener('click', () => {
    uploadModal.classList.add('hidden');
});

btnNewSession.addEventListener('click', () => {
    if (isProcessing) return;
    resetSession();
});

btnInterview.addEventListener('click', () => {
    interviewModal.classList.remove('hidden');
});

btnCloseInterview.addEventListener('click', closeInterviewModal);

btnStartInterview.addEventListener('click', () => {
    const prompt = buildInterviewPrompt();
    closeInterviewModal();
    // 进入面试模式
    isInInterviewMode = true;
    messageInput.placeholder = '🎤 语音作答或输入答案…';
    messageInput.focus();
    sendChat(prompt);
});

btnVoice.addEventListener('click', () => {
    if (!speechRecognition) return;
    if (isVoiceListening) {
        speechRecognition.stop();
        return;
    }
    speechRecognition.start();
});

// --- 初始化 ---
loadSessionList();
showWelcome();
initVoiceInput();
