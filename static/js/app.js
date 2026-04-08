/* ============================================
   多模态文档问答助手 - 前端逻辑
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
    showWelcome();
}

function showWelcome() {
    const welcome = document.createElement('div');
    welcome.className = 'message welcome';
    welcome.innerHTML = `<div class="message-content">
        <p>👋 你好！我是多模态文档问答助手。</p>
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
const btnUpload = document.getElementById('btn-upload');
const fileInput = document.getElementById('file-input');
const btnImage = document.getElementById('btn-image');
const imageInput = document.getElementById('image-input');
const imagePreview = document.getElementById('image-preview');
const previewImg = document.getElementById('preview-img');
const clearImageBtn = document.getElementById('clear-image');
const fileList = document.getElementById('file-list');
const statusIndicator = document.getElementById('status-indicator');
const uploadModal = document.getElementById('upload-modal');
const progressFill = document.getElementById('progress-fill');
const uploadStatusText = document.getElementById('upload-status-text');
const btnCloseModal = document.getElementById('btn-close-modal');
const btnNewSession = document.getElementById('btn-new-session');

// 简历分析相关 DOM
const btnResume = document.getElementById('btn-resume');
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
                jd_analysis: '🎯 岗位分析'
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
        resume_expert: '简历专家'
    };

    const items = agentSteps.map((step) => {
        const label = labels[step.agent] || step.agent || '处理中';
        const text = step.status === 'done' ? `已完成 ${label}` : `正在调用 ${label}`;
        return `<span class="agent-step ${step.status}">${escapeHtml(text)}</span>`;
    }).join('');

    return `<div class="agent-timeline">${items}</div>`;
}

function renderAiMessageContent({ routeType, fullAnswer, currentSources, agentSteps }) {
    const routeLabels = {
        kb: '📚 知识库',
        web: '🌐 网络',
        direct: '💬 直接回答',
        resume_analysis: '📋 简历分析',
        jd_analysis: '🎯 岗位分析'
    };

    const routeHtml = routeType ? `<span class="route-tag ${routeType}">${routeLabels[routeType] || routeType}</span><br>` : '';
    const timelineHtml = buildAgentTimelineHtml(agentSteps);
    const sourcesHtml = currentSources.length > 0 ? buildSourcesHtml(currentSources) : '';

    if (!fullAnswer) {
        return routeHtml + timelineHtml + sourcesHtml;
    }

    try {
        return routeHtml + timelineHtml + renderMarkdown(fullAnswer) + sourcesHtml;
    } catch (renderErr) {
        console.warn('Markdown 渲染异常，降级为纯文本:', renderErr);
        return routeHtml + timelineHtml + '<p>' + escapeHtml(fullAnswer) + '</p>' + sourcesHtml;
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
                        });
                        setStatus('生成中', 'processing');

                    } else if (payload.type === 'agent_start') {
                        agentSteps = agentSteps.filter(step => !(step.agent === payload.agent && step.status === 'running'));
                        agentSteps.push({ agent: payload.agent, status: 'running' });
                        const statusLabels = {
                            qa_flow: '正在检索知识库',
                            jd_expert: '正在分析岗位描述',
                            resume_expert: '正在评估简历'
                        };
                        setStatus(statusLabels[payload.agent] || '处理中', 'processing');
                        if (msgDiv) {
                            const contentEl = msgDiv.querySelector('.message-content');
                            contentEl.innerHTML = renderAiMessageContent({
                                routeType: currentRoute,
                                fullAnswer,
                                currentSources,
                                agentSteps,
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
                            });
                        }

                    } else if (payload.type === 'agent_cache_hit') {
                        setStatus('复用已有结果', 'processing');

                    } else if (payload.type === 'status') {
                        if (payload.content) {
                            setStatus(payload.content, 'processing');
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
                            });
                        }
                        if (payload.session_id && payload.session_id !== sessionId) {
                            sessionId = payload.session_id;
                            localStorage.setItem('agent_session_id', sessionId);
                        }
                        setStatus('就绪', 'ready');

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
                addFileToList(file.name, true);
            } else {
                addFileToList(file.name, false, data.detail);
            }
        } catch (err) {
            addFileToList(file.name, false, err.message);
        }
    }

    progressFill.style.width = '100%';
    uploadStatusText.textContent = `处理完成：${successCount}/${total} 成功`;
    btnCloseModal.classList.remove('hidden');
    setStatus('就绪', 'ready');
    isProcessing = false;
    console.log(`[Upload] 全部完成: ${successCount}/${total}`);
}

function addFileToList(name, success, error = null) {
    const hint = fileList.querySelector('.empty-hint');
    if (hint) hint.remove();

    const div = document.createElement('div');
    div.className = 'file-item';

    const ext = name.split('.').pop().toLowerCase();
    let icon = '📄';
    if (['pdf'].includes(ext)) icon = '📕';
    if (['png', 'jpg', 'jpeg'].includes(ext)) icon = '🖼️';
    if (['txt', 'md'].includes(ext)) icon = '📝';

    div.innerHTML = `
        <span class="file-icon">${icon}</span>
        <span class="file-name" title="${escapeHtml(name)}">${escapeHtml(name)}</span>
        <span class="file-status ${success ? '' : 'error'}">${success ? '✓' : '✕'}</span>
    `;

    if (error) {
        div.title = error;
    }

    fileList.appendChild(div);
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

clearImageBtn.addEventListener('click', clearImagePreview);

btnCloseModal.addEventListener('click', () => {
    uploadModal.classList.add('hidden');
});

btnNewSession.addEventListener('click', () => {
    if (isProcessing) return;
    resetSession();
});

// --- 初始化 ---
async function loadFileList() {
    try {
        const res = await fetch(`${API_BASE}/ingest/files`);
        if (res.ok) {
            const data = await res.json();
            if (data.files && data.files.length > 0) {
                fileList.querySelector('.empty-hint')?.remove();
                data.files.forEach(f => addFileToList(f.name, true));
            }
        }
    } catch (e) {}
}

loadFileList();
showWelcome();
