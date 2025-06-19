import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const messagesEndRef = useRef(null);
  const messageContainerRef = useRef(null);
  
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);
  
  const handleInputChange = (e) => {
    setInput(e.target.value);
  };
  
  const getStreamingResponse = async (message, messageId) => {
    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }
        
        buffer += decoder.decode(value, { stream: true });
        
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              setMessages(prev => 
                prev.map(msg => 
                  msg.id === messageId 
                    ? { 
                        ...msg, 
                        content: msg.content + (data.content || ''),
                        ...(data.finished ? { isStreaming: false } : {})
                      } 
                    : msg
                )
              );
              
              if (data.error) {
                throw new Error(data.error);
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }
      
      setMessages(prev => 
        prev.map(msg => 
          msg.id === messageId 
            ? { ...msg, isStreaming: false } 
            : msg
        )
      );
      
    } catch (error) {
      console.error('Error in streaming response:', error);
      setMessages(prev => 
        prev.map(msg => 
          msg.id === messageId 
            ? { ...msg, content: msg.content + '\n\n[连接中断] ' + error.message, isStreaming: false } 
            : msg
        )
      );
    }
  };
  
  const simulateStreamingResponse = async (message, messageId) => {
    try {
      const words = `这是一个模拟回应。目前无法连接到后端服务器，所以显示这条测试消息。您的问题是关于: "${message}"。请确保后端服务器正在运行，并且可以从前端访问API端点 /api/chat。`;
      const wordArray = words.split(' ');
      
      for (let i = 0; i < wordArray.length; i++) {
        setMessages(prev => 
          prev.map(msg => 
            msg.id === messageId 
              ? { 
                  ...msg, 
                  content: msg.content + (i > 0 ? ' ' : '') + wordArray[i],
                  ...(i === wordArray.length - 1 ? { isStreaming: false } : {})
                } 
              : msg
          )
        );
        
        await new Promise(resolve => setTimeout(resolve, 50));
      }
    } catch (error) {
      console.error('Error in simulation:', error);
      setMessages(prev => 
        prev.map(msg => 
          msg.id === messageId 
            ? { ...msg, content: '获取回应时出错', isStreaming: false } 
            : msg
        )
      );
    }
  };
  
  const handleSubmit = async (e) => {
    if (e) e.preventDefault();
    
    if (!input.trim() || isLoading) return;
    
    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toLocaleTimeString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setError(null);
    
    try {
      const aiMessageId = Date.now();
      setMessages(prev => [...prev, {
        id: aiMessageId,
        role: 'assistant',
        content: '',
        timestamp: new Date().toLocaleTimeString(),
        isStreaming: true
      }]);
      
      try {
        await getStreamingResponse(userMessage.content, aiMessageId);
      } catch (apiError) {
        console.warn('无法连接到后端API，使用模拟模式:', apiError);
        await simulateStreamingResponse(userMessage.content, aiMessageId);
      }
      
    } catch (err) {
      setError('连接服务器失败，请稍后再试');
      console.error('Error:', err);
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };
  
  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <header className="bg-blue-600 text-white p-4 shadow-md">
        <h1 className="text-xl font-bold">智慧海洋牧场 AI 助手</h1>
      </header>
      
      <div 
        ref={messageContainerRef}
        className="flex-1 p-4 overflow-y-auto"
      >
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500">
              <div className="text-5xl mb-4">🌊</div>
              <p className="text-lg">欢迎使用智慧海洋牧场 AI 助手</p>
              <p className="mt-2">请输入您的问题</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message, index) => (
              <div 
                key={index}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div 
                  className={`max-w-3/4 rounded-lg p-3 ${
                    message.role === 'user' 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-white border border-gray-200 text-gray-800'
                  }`}
                >
                  {message.role === 'user' ? (
                    <div className="whitespace-pre-wrap">{message.content}</div>
                  ) : (
                    <div className="markdown-content">
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                  )}
                  {message.isStreaming && (
                    <div className="mt-2">
                      <div className="flex space-x-1">
                        <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '-0.32s'}}></div>
                        <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '-0.16s'}}></div>
                        <div className="h-2 w-2 bg-gray-400 rounded-full animate-bounce"></div>
                      </div>
                    </div>
                  )}
                  <div className={`text-xs mt-1 ${message.role === 'user' ? 'text-blue-200' : 'text-gray-500'}`}>
                    {message.timestamp}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mx-4 mb-4">
          <p>{error}</p>
        </div>
      )}
      
      <div className="border-t border-gray-300 p-4 bg-white">
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
            placeholder="请输入您的问题..."
            className="flex-1 border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleSubmit}
            disabled={isLoading || !input.trim()}
            className={`px-4 py-2 rounded-lg font-medium ${
              isLoading || !input.trim()
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {isLoading ? '发送中...' : '发送'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default App;