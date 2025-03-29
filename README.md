# Tunisian Real Estate Law Chatbot

A sophisticated chatbot system designed to provide accurate information about Tunisian real estate law, built using modern AI technologies and best practices.

## 🌟 Features

- Multi-language support (English, French, Arabic)
- Context-aware responses using LangChain
- Vector-based information retrieval
- Comprehensive real estate law coverage
- Efficient data processing pipeline
- Web scraping capabilities for data collection

## 🛠️ Technology Stack

### Core Technologies
- **LangChain**: For building the chatbot pipeline
- **Ollama**: Using Mistral model for language processing
- **Chroma DB**: Vector database for efficient information retrieval
- **Selenium**: Web scraping for data collection
- **Python**: Primary programming language

### Key Libraries
- `langchain_chroma`: Vector store integration
- `langchain_ollama`: LLM integration
- `selenium`: Web automation
- `langdetect`: Language detection
- `json`: Data handling

## 📁 Project Structure

```
chatbot version 1/
├── chatbot.py           # Main chatbot implementation
├── chatbot.ipynb        # Jupyter notebook for development
├── scrape_chat.py       # Web scraping implementation
├── process_data.py      # Data processing utilities
├── new_data.py         # Data transformation logic
├── data.json           # Consolidated data
├── data_en.json        # English data
├── real_state_en.json  # Real estate specific data
├── registration_en.json # Registration related data
├── laws_en.json        # Legal framework data
└── chroma_db/          # Vector database storage
```

## 🔧 System Architecture

### 1. Data Collection Layer
- Automated web scraping from multiple sources
- Multi-language support (EN, FR, AR)
- Structured data extraction
- Error handling and retry mechanisms

### 2. Data Processing Layer
- JSON data consolidation
- Metadata preservation
- Structured data transformation
- Multi-source integration

### 3. Chatbot Core Layer
- Context-aware response generation
- Vector-based similarity search
- Language detection
- Efficient information retrieval

### 4. Model Architecture
- **Language Model**: Mistral (via Ollama)
- **Vector Store**: Chroma DB with Mistral embeddings
- **Processing Pipeline**: 
  1. Input Processing
  2. Language Detection
  3. Context Retrieval
  4. Response Generation
  5. Output Formatting

## 📊 System Components

### 1. Data Management
- Structured JSON storage
- Multi-language data organization
- Efficient data transformation
- Version control for different language versions

### 2. Processing Pipeline
- Document loading and processing
- Embedding generation
- Vector store management
- Response generation chain

### 3. Response Generation
- Custom prompt template
- Context-aware responses
- Error handling and fallback mechanisms
- Multi-language support

## 🎯 Use Cases

The chatbot can handle various types of queries about Tunisian real estate law, including:
- Co-ownership regulations
- Land classification
- Property registration procedures
- Legal frameworks
- General real estate queries

## 📈 Performance Characteristics

### Current Implementation
- Response Time: Optimized for real-time interaction
- Accuracy: Context-aware responses
- Scalability: Efficient vector storage
- Resource Usage: Optimized memory management

## 🔒 Security Features

1. **Input Validation**
   - Query length limits
   - Character validation
   - Special character handling

2. **Rate Limiting**
   - Request throttling
   - User quotas
   - Usage monitoring

3. **Data Protection**
   - Secure storage
   - Access control
   - Data encryption

## 📝 Usage Examples

```python
# Initialize chatbot
from chatbot import Chatbot
chatbot = Chatbot()

# Get response
response = chatbot.get_response("What are the requirements for property registration in Tunisia?")
print(response)
```


