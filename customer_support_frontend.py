#!/usr/bin/env python3
"""
Customer Support System Frontend - Multi-Agent Interface
======================================================

A comprehensive Gradio-based frontend serving as both MCP and A2A client for the
multi-agent customer support system. Provides an intuitive web interface for customer
support operations with real-time agent orchestration and vector RAG integration.

Key Features:
- Gradio web interface for customer support ticket submission
- MCP client integration for Gmail and Drive operations
- A2A client for multi-agent orchestration and task delegation
- Vector RAG system integration for FAQ and knowledge base search
- PDF upload and processing for knowledge base management
- Real-time conversation history and ticket tracking
- LangSmith tracing integration for observability

System Architecture:
- MCP Client: Communicates with Gmail and Drive MCP servers
- A2A Client: Orchestrates specialized support agents (Triage, Email, Documentation, Resolution, Escalation)
- Vector RAG: Provides intelligent FAQ and knowledge base search
- Gradio UI: Web interface for customer interactions and admin operations

Supported Operations:
- Submit customer support requests with automatic ticket creation
- Search knowledge base and FAQ with vector similarity
- Upload and process PDF documents for knowledge base
- View active tickets and conversation history
- Initialize and manage vector stores for FAQ/KB
- Real-time agent communication and task delegation

Environment Variables Required:
- MCP_GMAIL_URL: Gmail MCP server URL (default: http://localhost:8001)
- MCP_DRIVE_URL: Drive MCP server URL (default: http://localhost:8002)
- A2A_SERVER_URL: A2A agents server URL (default: http://localhost:8000)
- LANGSMITH_API_KEY: (Optional) LangSmith API key for tracing
- LANGSMITH_PROJECT: (Optional) LangSmith project name
- LANGSMITH_ENDPOINT: (Optional) LangSmith endpoint URL

Deployment Options:
1. Google Colab: Upload and run directly in Colab environment
2. Local Development: Run with `python customer_support_frontend.py`
3. Cloud Deployment: Deploy on platforms supporting Gradio apps

Usage:
    # Local development
    python customer_support_frontend.py
    
    # Google Colab
    !python customer_support_frontend.py

Author: Multi-Agent Customer Support System
Version: 2.0.0
"""

# Standard library imports
import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Third-party imports
import httpx
import gradio as gr

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# LANGSMITH INTEGRATION
# =============================================================================
# Optional LangSmith tracing for observability and debugging

try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    # Graceful fallback decorator if LangSmith not available
    def traceable(name=None):
        def decorator(func):
            return func
        return decorator
    LANGSMITH_AVAILABLE = False

# Setup LangSmith tracing
def setup_langsmith():
    """Setup LangSmith tracing if credentials are available"""
    if LANGSMITH_AVAILABLE:
        langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
        langsmith_project = os.getenv('LANGSMITH_PROJECT', 'MCP-A2A-Agents')
        langsmith_endpoint = os.getenv('LANGSMITH_ENDPOINT', 'https://api.smith.langchain.com')
        
        if langsmith_api_key:
            os.environ['LANGSMITH_TRACING'] = 'true'
            os.environ['LANGSMITH_ENDPOINT'] = langsmith_endpoint
            os.environ['LANGSMITH_API_KEY'] = langsmith_api_key
            os.environ['LANGSMITH_PROJECT'] = langsmith_project
            print(f"✅ LangSmith tracing enabled for frontend: {langsmith_project}")
        else:
            print("⚠️  LangSmith API key not found, frontend tracing disabled")
    else:
        print("⚠️  LangSmith not installed, frontend tracing disabled")

# Initialize LangSmith
setup_langsmith()

# =============================================================================
# MCP CLIENT
# =============================================================================
# Client for communicating with Gmail and Drive MCP servers

class MCPClient:
    """
    Model Context Protocol (MCP) client for Gmail and Drive server communication.
    
    Provides a unified interface for interacting with MCP Gmail and Drive servers,
    enabling AI agents to perform email operations, document management, and
    customer support ticket operations through standardized MCP protocols.
    
    Features:
    - Asynchronous HTTP client for MCP server communication
    - Tool invocation for Gmail operations (send, search, thread management)
    - Tool invocation for Drive operations (sheets, docs, vector RAG)
    - Resource retrieval for structured data access
    - Error handling and timeout management
    
    Attributes:
        gmail_url (str): URL of the Gmail MCP server
        drive_url (str): URL of the Drive MCP server
        client (httpx.AsyncClient): HTTP client for server communication
        
    Environment Variables:
        MCP_GMAIL_URL: Gmail MCP server URL (default: http://localhost:8001)
        MCP_DRIVE_URL: Drive MCP server URL (default: http://localhost:8002)
    """
    
    def __init__(self, gmail_url: str = None, drive_url: str = None):
        self.gmail_url = gmail_url or os.getenv('GMAIL_MCP_URL')
        self.drive_url = drive_url or os.getenv('DRIVE_MCP_URL')
        
        if not self.gmail_url:
            raise ValueError("Gmail MCP URL is required (set GMAIL_MCP_URL or pass gmail_url)")
        if not self.drive_url:
            raise ValueError("Drive MCP URL is required (set DRIVE_MCP_URL or pass drive_url)")
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def call_tool(self, server_url: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call MCP server tool"""
        try:
            response = await self.client.post(
                f"{server_url}/tools/{tool_name}",
                json=kwargs
            )
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def get_resource(self, server_url: str, resource_uri: str) -> str:
        """Get MCP server resource"""
        try:
            response = await self.client.get(
                f"{server_url}/resources",
                params={"uri": resource_uri}
            )
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

# =============================================================================
# A2A CLIENT
# =============================================================================
# Client for communicating with A2A multi-agent server

class A2AClient:
    """
    Agent-to-Agent (A2A) client using official A2A SDK protocol.
    
    Implements the correct A2A protocol for multi-agent orchestration and task delegation,
    following the official A2A SDK documentation and examples.
    
    Features:
    - Official A2A SDK protocol implementation
    - Agent card resolution and caching
    - Proper message structure with role, parts, and messageId
    - Error handling and timeout management
    
    Attributes:
        server_url (str): URL of the A2A agents server
        client (httpx.AsyncClient): HTTP client for server communication
        agent_card (dict): Cached agent card from server
        
    Environment Variables:
        A2A_SERVER_URL: A2A agents server URL (default: http://localhost:8000)
    """
    
    def __init__(self, server_url: str = None):
        self.server_url = server_url or os.getenv('A2A_SERVER_URL')
        
        if not self.server_url:
            raise ValueError("A2A Server URL is required (set A2A_SERVER_URL or pass server_url)")
        self.client = httpx.AsyncClient(timeout=30.0)
        self.agent_card = None
    
    async def _get_agent_card(self) -> Dict[str, Any]:
        """Fetch and cache agent card from A2A server"""
        if self.agent_card is not None:
            return self.agent_card
            
        try:
            response = await self.client.get(f"{self.server_url}/.well-known/agent.json")
            response.raise_for_status()
            self.agent_card = response.json()
            return self.agent_card
        except Exception as e:
            return {"error": f"Failed to fetch agent card: {str(e)}"}
    
    async def send_message(self, message: str, agent_type: str = "triage") -> Dict[str, Any]:
        """Send message to A2A agent using official A2A SDK protocol"""
        try:
            # Get agent card first
            agent_card = await self._get_agent_card()
            if "error" in agent_card:
                return agent_card
            
            # Generate unique message ID
            import uuid
            message_id = uuid.uuid4().hex
            request_id = str(uuid.uuid4())
            
            # Create A2A protocol message structure
            a2a_message = {
                "role": "user",
                "parts": [
                    {
                        "kind": "text",
                        "text": f"[{agent_type.upper()}] {message}"
                    }
                ],
                "messageId": message_id
            }
            
            # Create A2A SendMessageRequest payload
            request_payload = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "message/send",  # Fixed: Use forward slash, not dot
                "params": {
                    "message": a2a_message
                }
            }
            
            # Send request to A2A server
            response = await self.client.post(
                self.server_url,  # A2A servers handle JSONRPC at root
                json=request_payload,
                headers={"Content-Type": "application/json"}
            )
            
            # Check if response is successful
            response.raise_for_status()
            
            # Check if response has content
            if not response.content:
                return {"error": "Empty response from A2A server"}
            
            # Parse A2A response
            try:
                a2a_response = response.json()
                
                # Handle A2A error responses
                if "error" in a2a_response:
                    return {
                        "error": f"A2A error: {a2a_response['error'].get('message', 'Unknown error')}",
                        "code": a2a_response['error'].get('code', -1)
                    }
                
                # Extract result from A2A response
                result = a2a_response.get("result", {})
                
                # Extract message content from A2A response structure
                if isinstance(result, dict) and "message" in result:
                    message_result = result["message"]
                    if isinstance(message_result, dict) and "parts" in message_result:
                        # Extract text from parts
                        text_parts = []
                        for part in message_result["parts"]:
                            if part.get("kind") == "text":
                                text_parts.append(part.get("text", ""))
                        return {"message": " ".join(text_parts)}
                    else:
                        return {"message": str(message_result)}
                else:
                    return {"message": str(result)}
                
            except ValueError as json_error:
                return {
                    "error": f"Invalid JSON response: {str(json_error)}",
                    "raw_response": response.text,
                    "status_code": response.status_code
                }
                
        except httpx.HTTPStatusError as http_error:
            return {
                "error": f"HTTP error {http_error.response.status_code}: {http_error.response.text}",
                "status_code": http_error.response.status_code
            }
        except httpx.RequestError as req_error:
            return {
                "error": f"Request error: {str(req_error)}",
                "connection_error": True
            }
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    async def get_agent_card(self) -> Dict[str, Any]:
        """Get agent card information"""
        try:
            response = await self.client.get(f"{self.server_url}/.well-known/agent.json")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

# =============================================================================
# CLIENT INITIALIZATION
# =============================================================================
# Initialize MCP and A2A clients with environment-based configuration

# Initialize MCP client for Gmail and Drive operations
mcp_client = MCPClient()

# Initialize A2A client for multi-agent coordination
a2a_client = A2AClient()

# =============================================================================
# CUSTOMER SUPPORT SYSTEM
# =============================================================================
# Main system interface orchestrating MCP and A2A clients

class CustomerSupportSystem:
    """
    Main customer support system interface orchestrating multi-agent operations.
    
    Provides a unified interface for customer support operations, coordinating
    between MCP clients (Gmail/Drive) and A2A agents to deliver comprehensive
    customer service with intelligent ticket routing, knowledge base search,
    and automated response generation.
    
    Features:
    - Customer request processing with multi-agent coordination
    - Vector RAG integration for FAQ and knowledge base search
    - PDF upload and processing for knowledge base management
    - Conversation history tracking and ticket management
    - Real-time status updates and ticket monitoring
    - Automated email responses and escalation handling
    
    System Workflow:
    1. Customer submits request through Gradio interface
    2. Triage agent classifies and routes the request
    3. Resolution agent searches FAQ/KB using vector RAG
    4. Email agent composes and sends customer responses
    5. Documentation agent updates tickets and logs interactions
    6. Escalation agent handles complex issues requiring human intervention
    
    Attributes:
        conversation_history (List): Record of all customer interactions
        current_ticket_id (str): Active ticket identifier for tracking
        
    Integration Points:
        - MCP Gmail Server: Email operations and customer communication
        - MCP Drive Server: Document management and ticket logging
        - A2A Agents Server: Multi-agent orchestration and task delegation
        - Vector RAG System: Intelligent FAQ and knowledge base search
    """
    
    def __init__(self):
        self.conversation_history = []
        self.current_ticket_id = None
        self.a2a_client = a2a_client  # Initialize A2A client for vector store operations
    
    @traceable(name="process_customer_request")
    async def process_customer_request(self, customer_name: str, customer_email: str, request: str) -> str:
        """Process customer support request through A2A multi-agent system"""
        try:
            # Step 1: Send to A2A Triage Agent
            triage_response = await a2a_client.send_message(request, "triage")
            
            # Step 2: Based on triage, route to appropriate specialist
            triage_message = triage_response.get("message", "")
            if "ESCALATED" in str(triage_message) or "URGENT" in str(triage_message):
                specialist_response = await a2a_client.send_message(request, "escalation")
            elif "TECHNICAL" in str(triage_message) or "ERROR" in str(triage_message):
                specialist_response = await a2a_client.send_message(request, "resolution")
            elif "BILLING" in str(triage_message) or "PAYMENT" in str(triage_message):
                specialist_response = await a2a_client.send_message(request, "resolution")
            else:
                specialist_response = await a2a_client.send_message(request, "resolution")
            
            # Step 3: Log ticket via Documentation Agent
            ticket_data = {
                "action": "log_ticket",
                "customer_name": customer_name,
                "customer_email": customer_email,
                "description": request,
                "issue_type": "general",
                "priority": "medium"
            }
            
            doc_response = await a2a_client.send_message(json.dumps(ticket_data), "documentation")
            
            # Step 4: Send email response via Email Agent
            email_data = {
                "action": "send",
                "to": customer_email,
                "subject": "Support Response - Your Request",
                "body": f"Dear {customer_name},\n\nThank you for contacting support.\n\n{specialist_response.get('message', 'We are processing your request.')}\n\nBest regards,\nSupport Team"
            }
            
            email_response = await a2a_client.send_message(json.dumps(email_data), "email")
            
            # Extract meaningful content from A2A responses
            def extract_response_content(response_data):
                """Extract meaningful content from A2A agent response"""
                if isinstance(response_data, dict):
                    # Try different possible response fields
                    content = response_data.get('message') or response_data.get('content') or response_data.get('response')
                    if isinstance(content, dict):
                        # If content is a dict, try to extract solution or meaningful text
                        return content.get('solution') or content.get('result') or str(content)
                    elif isinstance(content, str):
                        return content
                    else:
                        return str(response_data)
                else:
                    return str(response_data)
            
            # Extract meaningful content from each A2A response
            triage_content = extract_response_content(triage_response)
            specialist_content = extract_response_content(specialist_response)
            doc_content = extract_response_content(doc_response)
            email_content = extract_response_content(email_response)
            
            # Compile response with extracted A2A content
            response = f"""
                🎯 **Triage Result:**
                {triage_content[:200] + '...' if len(triage_content) > 200 else triage_content}

                💡 **Solution:**
                {specialist_content[:500] + '...' if len(specialist_content) > 500 else specialist_content}

                📊 **Ticket Logged:**
                {doc_content[:100] + '...' if len(doc_content) > 100 else doc_content}

                📧 **Email Sent:**
                {email_content[:100] + '...' if len(email_content) > 100 else email_content}
                """
            
            # Add to conversation history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "customer": customer_name,
                "email": customer_email,
                "request": request,
                "response": response.strip()
            })
            
            # Automatically update FAQ vector store with new Q&A pair
            await self._auto_update_faq(request, specialist_content)
            
            return response.strip()
            
        except Exception as e:
            return f"❌ Error processing request: {str(e)}"
    
    async def _auto_update_faq(self, question: str, answer: str) -> None:
        """Automatically update FAQ vector store with new Q&A pair"""
        try:
            # Only add to FAQ if the answer is meaningful and not an error
            if answer and len(answer.strip()) > 20 and "❌" not in answer and "Error" not in answer:
                # Create FAQ entry in the format expected by the system
                faq_entry = {
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "timestamp": datetime.now().isoformat()
                }
                
                # Send to MCP Drive server to add to FAQ document and update vector store
                await mcp_client.call_tool(
                    mcp_client.drive_url,
                    "add_faq_entry",
                    **faq_entry
                )
                
                print(f"✅ Auto-added FAQ entry: {question[:50]}...")
        except Exception as e:
            print(f"⚠️ Failed to auto-update FAQ: {str(e)}")
            # Don't raise exception - FAQ update failure shouldn't break customer service
    
    def _classify_request_simple(self, request: str) -> dict:
        """Simple triage classification for direct MCP bypass"""
        request_lower = request.lower()
        
        # Escalation keywords
        if any(word in request_lower for word in ["urgent", "critical", "emergency", "down", "outage", "broken"]):
            return {
                "category": "escalation",
                "priority": "urgent",
                "confidence": 0.9
            }
        
        # Technical keywords
        elif any(word in request_lower for word in ["error", "bug", "crash", "performance", "slow", "api", "integration", "login", "password"]):
            priority = "high" if any(word in request_lower for word in ["crash", "error", "api"]) else "medium"
            return {
                "category": "technical",
                "priority": priority,
                "confidence": 0.8
            }
        
        # Billing keywords
        elif any(word in request_lower for word in ["billing", "payment", "invoice", "charge", "refund", "subscription"]):
            return {
                "category": "billing",
                "priority": "medium",
                "confidence": 0.8
            }
        
        # General support
        else:
            return {
                "category": "general",
                "priority": "low",
                "confidence": 0.6
            }
    
    @traceable(name="get_active_tickets")
    async def get_active_tickets(self) -> str:
        """Get current ticket status from sheets"""
        try:
            tickets_csv = await mcp_client.get_resource(mcp_client.drive_url, "sheet://tickets/active")
            return f"📊 **Active Tickets:**\n```\n{tickets_csv[:500]}...\n```"
        except Exception as e:
            return f"❌ Error retrieving tickets: {str(e)}"
    
    @traceable(name="search_knowledge_base")
    async def search_knowledge_base(self, query: str) -> str:
        """Search knowledge base for relevant information"""
        try:
            kb_result = await mcp_client.call_tool(
                mcp_client.drive_url,
                "search_knowledge_base",
                query=query
            )
            
            if isinstance(kb_result, list) and kb_result:
                results = "\n".join([f"• {doc.get('title', 'Untitled')}" for doc in kb_result[:5]])
                return f"🔍 **Knowledge Base Results:**\n{results}"
            else:
                return "🔍 **Knowledge Base:** No relevant documents found"
                
        except Exception as e:
            return f"❌ Error searching knowledge base: {str(e)}"



    async def initialize_kb_vectorstore(self, pdf_file_path: str) -> str:
        """Initialize KB vector store from uploaded PDF"""
        try:
            # Read the PDF file content and send it to backend
            if not pdf_file_path or not os.path.exists(pdf_file_path):
                return "❌ **KB Initialization Failed:**\nPDF file not found or invalid path"
            
            # Read PDF file as binary data
            with open(pdf_file_path, 'rb') as pdf_file:
                pdf_content = pdf_file.read()
            
            # Get filename from path
            filename = os.path.basename(pdf_file_path)
            
            # Send PDF content as multipart form data to backend
            files = {
                'pdf_file': (filename, pdf_content, 'application/pdf')
            }
            
            # Call A2A server to initialize KB vector store with file upload
            response = await self.a2a_client.client.post(
                f"{self.a2a_client.server_url}/vector-store/kb/initialize",
                files=files
            )
            result = response.json()
            
            if result.get("status") == "success":
                return f"✅ **KB Vector Store Initialized:**\n{result.get('message', 'Success')}"
            else:
                return f"❌ **KB Initialization Failed:**\n{result.get('message', 'Unknown error')}"
                
        except Exception as e:
            return f"❌ Error initializing KB vector store: {str(e)}"

    async def get_vectorstore_status(self) -> str:
        """Get vector store status"""
        try:
            # Get vector store status
            status_result = await mcp_client.call_tool(
                mcp_client.drive_url,
                "get_vectorstore_status"
            )
            return f"📊 **Vector Store Status:**\n{status_result}"
        except Exception as e:
            return f"❌ Error getting vector store status: {str(e)}"



# Initialize system
support_system = CustomerSupportSystem()

# Sync wrapper functions for Gradio with robust event loop handling
def run_async_safe(coro):
    """Safely run async coroutine in sync context"""
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, we need to run in a thread
            import concurrent.futures
            import threading
            
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=60)  # 60 second timeout
        else:
            # If no loop is running, we can use asyncio.run
            return asyncio.run(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)
    except Exception as e:
        return f"❌ Error running async operation: {str(e)}"

def process_request_sync(customer_name: str, customer_email: str, request_text: str) -> str:
    """Sync wrapper for processing support requests"""
    return run_async_safe(support_system.process_customer_request(customer_name, customer_email, request_text))

def search_knowledge_sync(query: str) -> str:
    """Sync wrapper for knowledge base search"""
    return run_async_safe(support_system.search_knowledge_base(query))



def initialize_kb_vectorstore_sync(pdf_file) -> str:
    """Sync wrapper for initializing KB vector store from uploaded PDF"""
    if pdf_file is None:
        return "❌ Please upload a PDF file first"
    return run_async_safe(support_system.initialize_kb_vectorstore(pdf_file.name))

def get_vectorstore_status_sync() -> str:
    """Sync wrapper for getting vector store status"""
    return run_async_safe(support_system.get_vectorstore_status())



def get_conversation_history() -> str:
    """Get formatted conversation history"""
    if not support_system.conversation_history:
        return "No conversations yet."
    
    history = ""
    for conv in support_system.conversation_history[-5:]:  # Last 5 conversations
        history += f"""
        **{conv['timestamp']}** - {conv['customer']}
        Request: {conv['request'][:100]}...
        Response: {conv['response'][:200]}...
        ---
        """
    return history

# Create Gradio Interface
def create_gradio_interface():
    """Create Gradio interface for customer support system"""
    
    with gr.Blocks(title="Multi-Agent Customer Support System", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 Multi-Agent Customer Support System")
        gr.Markdown("*Powered by MCP & A2A with specialized AI agents*")
        
        with gr.Tab("Customer Support"):
            gr.Markdown("## Submit Support Request")
            
            with gr.Row():
                with gr.Column():
                    customer_name = gr.Textbox(
                        label="Customer Name",
                        placeholder="Enter your name",
                        value="John Doe"
                    )
                    customer_email = gr.Textbox(
                        label="Email Address", 
                        placeholder="your.email@example.com",
                        value="john.doe@example.com"
                    )
                
                with gr.Column():
                    request_text = gr.Textbox(
                        label="Support Request",
                        placeholder="Describe your issue or question...",
                        lines=4,
                        value="I'm having trouble logging into my account. The password reset isn't working."
                    )
            
            submit_btn = gr.Button("Submit Request", variant="primary")
            response_output = gr.Textbox(
                label="System Response",
                lines=10,
                interactive=False
            )
            
            submit_btn.click(
                fn=process_request_sync,
                inputs=[customer_name, customer_email, request_text],
                outputs=response_output
            )
        
        # with gr.Tab("Ticket Management"):
        #     gr.Markdown("## Ticket Status & Management")
            
        #     status_btn = gr.Button("Get Active Tickets")
        #     status_output = gr.Textbox(
        #         label="Active Tickets",
        #         lines=8,
        #         interactive=False
        #     )
            
        #     status_btn.click(
        #         fn=get_status_sync,
        #         outputs=status_output
        #     )
        
        with gr.Tab("Vector Store Management"):
            gr.Markdown("## RAG Vector Store Management")
            
            gr.Markdown("""
            ### 🤖 **Automatic FAQ Management**
            The FAQ vector store is automatically created and updated as new customer questions are processed.
            No manual intervention required - the system learns from each interaction!
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Knowledge Base Vector Store")
                    pdf_upload = gr.File(
                        label="Upload Knowledge Base PDF",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    init_kb_btn = gr.Button("Initialize KB Vector Store", variant="primary")
            
            status_btn = gr.Button("Get Vector Store Status")
            vector_output = gr.Textbox(
                label="Vector Store Operations",
                lines=8,
                interactive=False
            )
            
            # Event handlers
            init_kb_btn.click(
                fn=initialize_kb_vectorstore_sync,
                inputs=[pdf_upload],
                outputs=vector_output
            )
            
            status_btn.click(
                fn=get_vectorstore_status_sync,
                outputs=vector_output
            )
        
        with gr.Tab("System Status"):
            gr.Markdown("## System Status & Conversation History")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""
                    ### 🔧 **MCP Servers**
                    - **Gmail Server**: Port 8001
                    - **Drive Server**: Port 8002
                    
                    ### 🤖 **A2A Agents**
                    - **Triage Agent**: Classification & routing
                    - **Email Agent**: Communication
                    - **Documentation Agent**: Ticket logging
                    - **Resolution Agent**: Solution generation
                    - **Escalation Agent**: Complex issues
                    """)
                    
                    history_btn = gr.Button("Get Conversation History")
                
                with gr.Column():
                    history_output = gr.Textbox(
                        label="Recent Conversations",
                        lines=10,
                        interactive=False,
                        value=get_conversation_history()
                    )
                    
                    refresh_btn = gr.Button("Refresh History")
            
            # Connect buttons to functions
            history_btn.click(
                fn=get_conversation_history,
                outputs=history_output
            )
            refresh_btn.click(
                fn=get_conversation_history,
                outputs=history_output
            )
    
    return interface

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Customer Support System Frontend...")
    print("🔗 Connecting to:")
    print(f"   • MCP Gmail Server: {mcp_client.gmail_url}")
    print(f"   • MCP Drive Server: {mcp_client.drive_url}")
    print(f"   • A2A Agent Server: {a2a_client.server_url}")
    
    port = int(os.environ.get("PORT", 7860))
    
    # Create and launch interface
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True
    )
