import React, { useState, useEffect, useRef } from 'react';
import { Send, Upload, BookOpen, Settings, Loader, Check, X, Video, Database, Cpu } from 'lucide-react';

// Get API URL - works in both build and artifact preview
const getApiUrl = () => {
  // In production build, this will be replaced
  if (typeof window !== 'undefined' && window.REACT_APP_API_URL) {
    return window.REACT_APP_API_URL;
  }
  // Default for local development and artifact preview
  //return 'http://localhost:8000';
  return 'http://80.225.218.170:8000';
};

const API_URL = getApiUrl();

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [isConnected, setIsConnected] = useState(false);
  const [healthData, setHealthData] = useState(null);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      const data = await response.json();
      setHealthData(data);
      setIsConnected(true);
    } catch (error) {
      console.error('Health check failed:', error);
      setIsConnected(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50">
      {/* Header */}
      <header className="bg-white shadow-lg border-b-4 border-indigo-600">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <BookOpen className="w-10 h-10 text-indigo-600" />
              <div>
                <h1 className="text-3xl font-bold text-gray-800">CBSE RAG System</h1>
                <p className="text-sm text-gray-600">AI-Powered Learning Assistant</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className={`flex items-center space-x-2 px-3 py-2 rounded-lg ${
                isConnected ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
              }`}>
                <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`}></div>
                <span className="text-sm font-semibold">
                  {isConnected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
          </div>
          
          {/* Tabs */}
          <div className="flex space-x-2 mt-4">
            <TabButton active={activeTab === 'chat'} onClick={() => setActiveTab('chat')}>
              💬 Chat
            </TabButton>
            <TabButton active={activeTab === 'upload'} onClick={() => setActiveTab('upload')}>
              📚 Upload
            </TabButton>
            <TabButton active={activeTab === 'status'} onClick={() => setActiveTab('status')}>
              📊 Status
            </TabButton>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'chat' && <ChatTab />}
        {activeTab === 'upload' && <UploadTab />}
        {activeTab === 'status' && <StatusTab healthData={healthData} />}
      </main>
    </div>
  );
}

function TabButton({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-2 rounded-lg font-semibold transition-all ${
        active
          ? 'bg-indigo-600 text-white shadow-lg'
          : 'bg-white text-gray-700 hover:bg-gray-100'
      }`}
    >
      {children}
    </button>
  );
}

function ChatTab() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [grade, setGrade] = useState('10');
  const [subject, setSubject] = useState('Mathematics');
  const [model, setModel] = useState('openai/gpt-4-turbo');
  const messagesEndRef = useRef(null);

  const subjects = ['Mathematics', 'Science', 'Biology', 'Physics', 'Chemistry', 
                   'English', 'Hindi', 'Social Studies', 'History', 'Geography', 'Civics'];
  
  const models = [
    { value: 'openai/gpt-4-turbo', label: 'GPT-4 Turbo (Best)' },
    { value: 'openai/gpt-3.5-turbo', label: 'GPT-3.5 (Fast)' },
    { value: 'anthropic/claude-3-opus', label: 'Claude 3 Opus' },
    { value: 'anthropic/claude-3-sonnet', label: 'Claude 3 Sonnet' },
    { value: 'google/gemini-pro', label: 'Gemini Pro' },
    { value: 'meta-llama/llama-3-70b', label: 'Llama 3 70B' }
  ];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage = {
      type: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: input,
          grade: parseInt(grade),
          subject: subject,
          llm_model: model,
          use_reranking: true,
          use_hyde: true
        })
      });

      const data = await response.json();

      const assistantMessage = {
        type: 'assistant',
        content: data.answer,
        sources: data.sources,
        metadata: data.metadata,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      const errorMessage = {
        type: 'error',
        content: `Error: ${error.message}. Make sure the backend is running.`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
      {/* Settings Sidebar */}
      <div className="lg:col-span-1 space-y-4">
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="font-bold text-gray-800 mb-4 flex items-center">
            <Settings className="w-5 h-5 mr-2" />
            Settings
          </h3>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">Grade</label>
              <select
                value={grade}
                onChange={(e) => setGrade(e.target.value)}
                className="w-full px-3 py-2 border-2 border-gray-200 rounded-lg focus:border-indigo-500 focus:outline-none"
              >
                {Array.from({length: 12}, (_, i) => i + 1).map(g => (
                  <option key={g} value={g}>Grade {g}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">Subject</label>
              <select
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                className="w-full px-3 py-2 border-2 border-gray-200 rounded-lg focus:border-indigo-500 focus:outline-none"
              >
                {subjects.map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">AI Model</label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full px-3 py-2 border-2 border-gray-200 rounded-lg focus:border-indigo-500 focus:outline-none text-sm"
              >
                {models.map(m => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="mt-6 p-4 bg-indigo-50 rounded-lg">
            <p className="text-xs text-indigo-800">
              💡 <strong>Tip:</strong> Use GPT-4 for best quality, GPT-3.5 for speed, or Gemini Pro for cost savings.
            </p>
          </div>
        </div>

        {/* Quick Examples */}
        <div className="bg-white rounded-xl shadow-lg p-6">
          <h3 className="font-bold text-gray-800 mb-3">Quick Examples</h3>
          <div className="space-y-2">
            <button
              onClick={() => setInput("Explain the quadratic formula with an example")}
              className="w-full text-left text-sm px-3 py-2 bg-gray-50 hover:bg-gray-100 rounded-lg transition"
            >
              📐 Quadratic formula
            </button>
            <button
              onClick={() => setInput("What is photosynthesis?")}
              className="w-full text-left text-sm px-3 py-2 bg-gray-50 hover:bg-gray-100 rounded-lg transition"
            >
              🌱 Photosynthesis
            </button>
            <button
              onClick={() => setInput("When did India gain independence?")}
              className="w-full text-left text-sm px-3 py-2 bg-gray-50 hover:bg-gray-100 rounded-lg transition"
            >
              🇮🇳 Independence
            </button>
          </div>
        </div>
      </div>

      {/* Chat Area */}
      <div className="lg:col-span-3">
        <div className="bg-white rounded-xl shadow-2xl flex flex-col h-[calc(100vh-250px)]">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.length === 0 ? (
              <div className="text-center py-20">
                <BookOpen className="w-20 h-20 text-indigo-300 mx-auto mb-4" />
                <h3 className="text-xl font-bold text-gray-700 mb-2">Welcome to CBSE RAG!</h3>
                <p className="text-gray-500">Ask any question about your curriculum</p>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <MessageBubble key={idx} message={msg} />
              ))
            )}

            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-2xl px-6 py-4">
                  <div className="flex items-center space-x-3">
                    <Loader className="w-5 h-5 text-indigo-600 animate-spin" />
                    <span className="text-gray-600">Thinking...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="border-t-2 border-gray-200 p-4 bg-gray-50">
            <div className="flex space-x-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !isLoading && handleSend()}
                placeholder="Ask about your curriculum..."
                className="flex-1 px-4 py-3 border-2 border-gray-300 rounded-xl focus:border-indigo-500 focus:outline-none"
                disabled={isLoading}
              />
              <button
                onClick={handleSend}
                disabled={isLoading || !input.trim()}
                className="px-6 py-3 bg-indigo-600 text-white rounded-xl hover:bg-indigo-700 transition disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }) {
  const isUser = message.type === 'user';
  const isError = message.type === 'error';

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-2xl rounded-2xl px-6 py-4 ${
        isUser 
          ? 'bg-indigo-600 text-white' 
          : isError
          ? 'bg-red-100 text-red-800 border-2 border-red-300'
          : 'bg-gray-100 text-gray-800'
      }`}>
        <p className="whitespace-pre-wrap">{message.content}</p>

        {message.sources && message.sources.length > 0 && (
          <div className="mt-4 pt-4 border-t border-gray-300">
            <p className="text-sm font-semibold mb-2">📚 Sources:</p>
            {message.sources.map((source, i) => (
              <div key={i} className="text-xs mb-2 bg-white bg-opacity-30 rounded p-2">
                <div className="font-semibold">
                  {source.metadata?.chapter || 'Unknown Chapter'}
                </div>
                <div className="text-opacity-80">
                  Relevance: {(source.score * 100).toFixed(1)}%
                </div>
              </div>
            ))}
          </div>
        )}

        {message.metadata && (
          <div className="mt-3 pt-3 border-t border-gray-300">
            <p className="text-xs opacity-75">
              Model: {message.metadata.llm_model} | 
              Sources: {message.metadata.num_sources}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function UploadTab() {
  const [file, setFile] = useState(null);
  const [subject, setSubject] = useState('Mathematics');
  const [grade, setGrade] = useState('10');
  const [chapter, setChapter] = useState('');
  const [topic, setTopic] = useState('');
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState(null);
  const [documents, setDocuments] = useState([]);

  const subjects = ['Mathematics', 'Science', 'Biology', 'Physics', 'Chemistry', 
                   'English', 'Hindi', 'Social Studies', 'History', 'Geography'];

  useEffect(() => {
    fetchDocuments();
  }, []);

  const fetchDocuments = async () => {
    try {
      const response = await fetch(`${API_URL}/documents`);
      const data = await response.json();
      setDocuments(data.documents || []);
    } catch (error) {
      console.error('Failed to fetch documents:', error);
    }
  };

  const handleUpload = async () => {
    if (!file || !chapter) {
      setMessage({ type: 'error', text: 'Please select a file and enter chapter name' });
      return;
    }

    setUploading(true);
    setMessage(null);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('subject', subject);
    formData.append('grade', grade);
    formData.append('chapter', chapter);
    formData.append('topic', topic);

    try {
      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      setMessage({ type: 'success', text: data.message || 'Document uploaded successfully!' });
      setFile(null);
      setChapter('');
      setTopic('');
      fetchDocuments();
    } catch (error) {
      setMessage({ type: 'error', text: `Upload failed: ${error.message}` });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Upload Form */}
      <div className="bg-white rounded-xl shadow-xl p-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
          <Upload className="w-7 h-7 mr-3 text-indigo-600" />
          Upload Document
        </h2>

        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">Subject *</label>
              <select
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-indigo-500 focus:outline-none"
              >
                {subjects.map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-semibold text-gray-700 mb-2">Grade *</label>
              <select
                value={grade}
                onChange={(e) => setGrade(e.target.value)}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-indigo-500 focus:outline-none"
              >
                {Array.from({length: 12}, (_, i) => i + 1).map(g => (
                  <option key={g} value={g}>Grade {g}</option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Chapter Name *</label>
            <input
              type="text"
              value={chapter}
              onChange={(e) => setChapter(e.target.value)}
              placeholder="e.g., Quadratic Equations"
              className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-indigo-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="block text-sm font-semibold text-gray-700 mb-2">Topic (Optional)</label>
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="e.g., Solving equations"
              className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-indigo-500 focus:outline-none"
            />
          </div>

          <div className="border-4 border-dashed border-indigo-200 rounded-xl p-8 text-center hover:border-indigo-400 transition">
            <input
              type="file"
              accept=".pdf"
              onChange={(e) => setFile(e.target.files[0])}
              className="hidden"
              id="file-upload"
            />
            <label htmlFor="file-upload" className="cursor-pointer">
              {file ? (
                <div>
                  <Check className="w-12 h-12 text-green-500 mx-auto mb-3" />
                  <p className="font-semibold text-gray-700">{file.name}</p>
                  <p className="text-sm text-gray-500 mt-1">
                    {(file.size / 1024).toFixed(2)} KB
                  </p>
                </div>
              ) : (
                <div>
                  <Upload className="w-12 h-12 text-indigo-400 mx-auto mb-3" />
                  <p className="font-semibold text-gray-700">Click to upload PDF</p>
                  <p className="text-sm text-gray-500 mt-1">Maximum file size: 10MB</p>
                </div>
              )}
            </label>
          </div>

          {message && (
            <div className={`p-4 rounded-lg ${
              message.type === 'success'
                ? 'bg-green-50 text-green-800 border-2 border-green-200'
                : 'bg-red-50 text-red-800 border-2 border-red-200'
            }`}>
              {message.text}
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={uploading || !file || !chapter}
            className="w-full py-4 bg-indigo-600 text-white rounded-xl font-bold text-lg hover:bg-indigo-700 transition disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
          >
            {uploading ? (
              <>
                <Loader className="w-5 h-5 animate-spin" />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <Upload className="w-5 h-5" />
                <span>Upload & Process</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Documents List */}
      <div className="bg-white rounded-xl shadow-xl p-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
          <Database className="w-7 h-7 mr-3 text-indigo-600" />
          Uploaded Documents ({documents.length})
        </h2>

        <div className="space-y-3 max-h-[600px] overflow-y-auto">
          {documents.length === 0 ? (
            <div className="text-center py-12 text-gray-400">
              <Database className="w-16 h-16 mx-auto mb-3" />
              <p>No documents uploaded yet</p>
            </div>
          ) : (
            documents.map((doc, idx) => (
              <div key={idx} className="border-2 border-gray-200 rounded-lg p-4 hover:border-indigo-300 transition">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-800">{doc.filename}</h4>
                    <div className="flex gap-3 text-sm text-gray-600 mt-2">
                      <span className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded">
                        Grade {doc.grade}
                      </span>
                      <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded">
                        {doc.subject}
                      </span>
                      <span className="px-2 py-1 bg-green-100 text-green-700 rounded">
                        {doc.chapter}
                      </span>
                    </div>
                    {doc.upload_date && (
                      <p className="text-xs text-gray-500 mt-2">
                        Uploaded: {new Date(doc.upload_date).toLocaleDateString()}
                      </p>
                    )}
                  </div>
                  <div className="ml-4">
                    {doc.status === 'processed' ? (
                      <Check className="w-6 h-6 text-green-500" />
                    ) : (
                      <Loader className="w-6 h-6 text-indigo-500 animate-spin" />
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

function StatusTab({ healthData }) {
  return (
    <div className="space-y-6">
      {/* System Status */}
      <div className="bg-white rounded-xl shadow-xl p-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
          <Cpu className="w-7 h-7 mr-3 text-indigo-600" />
          System Status
        </h2>

        {healthData ? (
          <div className="space-y-6">
            <div className="flex items-center space-x-4 p-4 bg-green-50 border-2 border-green-200 rounded-lg">
              <Check className="w-8 h-8 text-green-600" />
              <div>
                <p className="font-bold text-green-800">System Online</p>
                <p className="text-sm text-green-600">
                  Environment: {healthData.environment}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <ServiceCard 
                label="Vector DB" 
                status={healthData.services?.vector_db}
                icon={<Database className="w-5 h-5" />}
              />
              <ServiceCard 
                label="LLM" 
                status={healthData.services?.llm}
                icon={<Cpu className="w-5 h-5" />}
              />
              <ServiceCard 
                label="Embeddings" 
                status={healthData.services?.embeddings}
                icon={<BookOpen className="w-5 h-5" />}
              />
              <ServiceCard 
                label="Processing" 
                status={healthData.services?.processing}
                icon={<Loader className="w-5 h-5" />}
              />
            </div>

            {healthData.stats && (
              <div className="grid grid-cols-3 gap-4 mt-6">
                <StatCard
                  label="Documents"
                  value={healthData.stats.total_documents || 0}
                  icon="📚"
                />
                <StatCard
                  label="Total Queries"
                  value={healthData.stats.total_queries || 0}
                  icon="💬"
                />
                <StatCard
                  label="Avg Response"
                  value={`${healthData.stats.avg_response_time || 0}s`}
                  icon="⚡"
                />
              </div>
            )}

            <div className="mt-6 p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">
                <strong>Backend URL:</strong> {API_URL}
              </p>
              <p className="text-sm text-gray-600 mt-1">
                <strong>Last Updated:</strong> {new Date(healthData.timestamp).toLocaleString()}
              </p>
            </div>
          </div>
        ) : (
          <div className="text-center py-12">
            <X className="w-16 h-16 text-red-400 mx-auto mb-4" />
            <p className="text-gray-600">Unable to connect to backend</p>
            <p className="text-sm text-gray-500 mt-2">Check if the server is running at {API_URL}</p>
          </div>
        )}
      </div>

      {/* Features */}
      <div className="bg-white rounded-xl shadow-xl p-8">
        <h2 className="text-2xl font-bold text-gray-800 mb-6">Active Features</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <FeatureBadge icon="🔍" label="Hybrid Search" active={true} />
          <FeatureBadge icon="🎯" label="HyDE Expansion" active={true} />
          <FeatureBadge icon="📊" label="Reranking" active={true} />
          <FeatureBadge icon="🧩" label="Contextual Chunks" active={true} />
          <FeatureBadge icon="🌐" label="Multi-LLM" active={true} />
          <FeatureBadge icon="📝" label="Source Citations" active={true} />
        </div>
      </div>
    </div>
  );
}

function ServiceCard({ label, status, icon }) {
  const isHealthy = status === 'healthy' || status === 'connected' || status === 'ready';
  
  return (
    <div className={`p-4 rounded-lg border-2 ${
      isHealthy 
        ? 'bg-green-50 border-green-200' 
        : 'bg-yellow-50 border-yellow-200'
    }`}>
      <div className="flex items-center justify-between mb-2">
        <span className={isHealthy ? 'text-green-600' : 'text-yellow-600'}>
          {icon}
        </span>
        {isHealthy ? (
          <Check className="w-4 h-4 text-green-600" />
        ) : (
          <X className="w-4 h-4 text-yellow-600" />
        )}
      </div>
      <p className="text-xs font-semibold text-gray-700">{label}</p>
      <p className={`text-xs ${isHealthy ? 'text-green-600' : 'text-yellow-600'}`}>
        {status || 'Unknown'}
      </p>
    </div>
  );
}

function StatCard({ label, value, icon }) {
  return (
    <div className="bg-gradient-to-br from-indigo-500 to-purple-600 text-white p-6 rounded-xl shadow-lg">
      <div className="text-3xl mb-2">{icon}</div>
      <p className="text-3xl font-bold">{value}</p>
      <p className="text-sm opacity-90">{label}</p>
    </div>
  );
}

function FeatureBadge({ icon, label, active }) {
  return (
    <div className={`flex items-center space-x-3 p-3 rounded-lg ${
      active ? 'bg-indigo-50 border-2 border-indigo-200' : 'bg-gray-50 border-2 border-gray-200'
    }`}>
      <span className="text-2xl">{icon}</span>
      <div className="flex-1">
        <p className="text-sm font-semibold text-gray-800">{label}</p>
        <p className={`text-xs ${active ? 'text-indigo-600' : 'text-gray-500'}`}>
          {active ? 'Active' : 'Inactive'}
        </p>
      </div>
      {active && <Check className="w-5 h-5 text-indigo-600" />}
    </div>
  );
}


export default App;



