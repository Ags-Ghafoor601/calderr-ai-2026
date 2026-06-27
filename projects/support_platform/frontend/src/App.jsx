import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { Send, Bot, User, Sparkles } from 'lucide-react'
import './index.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [sessionId, setSessionId] = useState(null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, isLoading])

  // Start session on load
  useEffect(() => {
    const initSession = async () => {
      try {
        const res = await axios.post(`${API_URL}/sessions`)
        setSessionId(res.data.session_id)
        setMessages([
          { role: 'agent', content: 'Hello! I am your AI Support Assistant. How can I help you today?' }
        ])
      } catch (error) {
        console.error('Failed to start session:', error)
      }
    }
    initSession()
  }, [])

  const handleSend = async () => {
    if (!input.trim() || !sessionId || isLoading) return

    const userMessage = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage, time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) }])
    setIsLoading(true)

    try {
      const res = await axios.post(`${API_URL}/messages`, {
        session_id: sessionId,
        message: userMessage
      })
      
      setMessages(prev => [...prev, { 
        role: 'agent', 
        content: res.data.response,
        time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) 
      }])
    } catch (error) {
      console.error('Failed to send message:', error)
      setMessages(prev => [...prev, { 
        role: 'agent', 
        content: 'I am sorry, I am having trouble connecting to my servers right now.' 
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <>
      <div className="bg-orb bg-orb-1"></div>
      <div className="bg-orb bg-orb-2"></div>
      
      <div className="app-container">
        {/* Header */}
        <header className="header">
          <div className="header-icon">
            <Sparkles className="text-white" size={24} />
          </div>
          <div className="header-title">
            <h1>Nexus Support</h1>
            <p>AI-Powered Assistant</p>
          </div>
        </header>

        {/* Chat Area */}
        <main className="chat-container">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message-wrapper ${msg.role}`}>
              <div className="message-bubble">
                {msg.content}
              </div>
              {msg.time && <div className="message-time">{msg.time}</div>}
            </div>
          ))}
          
          {isLoading && (
            <div className="message-wrapper agent">
              <div className="message-bubble">
                <div className="typing-indicator">
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                  <div className="typing-dot"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </main>

        {/* Input Area */}
        <footer className="input-container">
          <div className="input-box">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message here..."
              rows={1}
            />
            <button 
              className="send-button" 
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
            >
              <Send size={20} />
            </button>
          </div>
        </footer>
      </div>
    </>
  )
}

export default App
