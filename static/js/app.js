/* ============================================
   多模态文档问答助手 - 前端逻辑
   阶段 3：新增简历分析功能
   ============================================ */

const API_BASE = '';  // 同源部署，留空

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

let pendingImageBase64 = null;
let pendingResumeFile = null;
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

    // 代码块（必须先处理，避免内部被其他规则干扰）
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
    });

    // 行内代码
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // 链接
    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

    // 表格
    html = html.replace(/^(\|.+\|)\n(\|[-| :]+\|)\n((?:\|.+\|\n?)+)/gm, (match, header, sep, body) => {
        const headers = header.split('|').filter(c => c.trim()).map(c => `<th>${c.trim()}</th>`).join('');
        const rows = body.trim().split('\n').map(row => {
            const cells = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('');
            return `<tr>${cells}</tr>`;
        }).join('');
        return `<table><thead><tr>${headers}</tr></thead><tbody>${rows}</tbody></table>`;
    });

    // 标题
    html = html.replace(/^#### (.+)$/gm, '<h5>$1</h5>');
    html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^# (.+)$/gm, '<h2>$1</h2>');

    // 无序列表
    html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
    // 有序列表
    html = html.replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>');
    // 包裹连续的 li 为 ul
    html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => `<ul>${match}</ul>`);

    // 粗体 / 斜体
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // 水平线
    html = html.replace(/^---$/gm, '<hr>');

    // 段落（处理连续换行）
    html = html.replace(/\n\n+/g, '</p><p>');
    html = html.replace(/\n/g, '<br>');

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
                resume_analysis: '📋 简历分析'
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
                        setStatus('生成中', 'processing');

                    } else if (payload.type === 'sources') {
                        // 检索来源
                        currentSources = payload.sources || [];
                        // 更新消息气泡的 sources
                        if (msgDiv) {
                            const sourcesHtml = buildSourcesHtml(currentSources);
                            const existingContent = msgDiv.querySelector('.message-content').innerHTML;
                            msgDiv.querySelector('.message-content').innerHTML = existingContent + sourcesHtml;
                        }

                    } else if (payload.type === 'token') {
                        // 增量文本
                        fullAnswer += payload.content;
                        if (msgDiv) {
                            // 增量渲染 Markdown
                            const contentEl = msgDiv.querySelector('.message-content');
                            const sourcesHtml = currentSources.length > 0 ? buildSourcesHtml(currentSources) : '';
                            contentEl.innerHTML = renderMarkdown(fullAnswer) + sourcesHtml;
                            scrollToBottom();
                        }

                    } else if (payload.type === 'done') {
                        // 完成
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

// --- 简历分析 ---
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

        const data = await res.json();
        removeTypingIndicator();

        if (res.ok) {
            if (data.session_id && data.session_id !== sessionId) {
                sessionId = data.session_id;
                localStorage.setItem('agent_session_id', sessionId);
            }
            // 简历分析用 Markdown 渲染
            appendMessage('ai', data.answer, data.sources, 'resume_analysis', true);
        } else {
            appendMessage('ai', `❌ 简历分析失败：${data.detail || '未知错误'}`);
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
