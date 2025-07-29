# AI PDF Assistant

An intelligent PDF assistant built with the Phi framework that can answer questions about PDF documents using RAG (Retrieval-Augmented Generation). The assistant processes PDF content, creates embeddings, and provides conversational AI interactions.

## Features

- **PDF Knowledge Base**: Automatically processes and indexes PDF documents
- **Conversational AI**: Chat with your PDFs using natural language
- **Local Embeddings**: Uses Sentence Transformers for free, offline embeddings
- **Chat History**: Maintains conversation context across sessions
- **SQLite Storage**: No complex database setup required
- **Groq Integration**: Fast inference using Groq's Llama models

## Tech Stack

- **Framework**: Phi (phidata)
- **AI Model**: Groq (Llama 3.3 70B)
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: SQLite with vector search
- **PDF Processing**: PyPDF
- **CLI Interface**: Typer

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/pdf-assistant.git
cd pdf-assistant
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**:
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
PHI_API_KEY=your_phi_api_key_here
```

## API Keys Setup

### Get Groq API Key (Free)
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Generate an API key
4. Add to your `.env` file

### Get Phi API Key (Optional)
1. Visit [Phi Data](https://phidata.com/)
2. Sign up and get your API key
3. Add to your `.env` file

## Usage

### Basic Usage
```bash
python pdf_assistant.py
```

### Start New Conversation
```bash
python pdf_assistant.py --new
```

### Use with Specific User
```bash
python pdf_assistant.py --user john_doe
```

## Project Structure

```
pdf-assistant/
├── pdf_assistant.py          # Main application file
├── requirements.txt          # Python dependencies
├── .env                     # Environment variables (not tracked)
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
├── pdf_assistant.db        # SQLite database (auto-created)
└── venv/                   # Virtual environment (not tracked)
```

## How It Works

1. **PDF Processing**: The assistant downloads and processes the specified PDF
2. **Embedding Creation**: Text chunks are converted to vectors using local embeddings
3. **Vector Storage**: Embeddings are stored in SQLite for fast retrieval
4. **Query Processing**: User questions are embedded and matched against stored vectors
5. **Response Generation**: Relevant context is sent to Groq's Llama model for response generation

## Customization

### Change PDF Source
Edit the `urls` parameter in `pdf_assistant.py`:
```python
knowledge_base = PDFUrlKnowledgeBase(
    urls=["your_pdf_url_here"],
    # ... rest of config
)
```

### Change AI Model
Modify the model in the Assistant configuration:
```python
model=Groq(id="mixtral-8x7b-32768", api_key=groq_api_key)
```

### Use Different Embedding Model
Change the embedder in the vector database:
```python
embedder=SentenceTransformerEmbedder(model="all-mpnet-base-v2")
```

## Example Conversations

```
User: What are the main ingredients in Thai curry?
Assistant: Based on the Thai recipes document, the main ingredients in Thai curry typically include...

User: How do I make pad thai?
Assistant: According to the recipes, here's how to make pad thai...
```

## Troubleshooting

### Common Issues

**API Key Errors**:
- Ensure your `.env` file is in the project root
- Check that API keys are valid and have sufficient quota

**Database Issues**:
- Delete `pdf_assistant.db` to reset the database
- Ensure you have write permissions in the project directory

**Memory Issues**:
- Try using a smaller embedding model
- Process smaller PDF files

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## ⚠️ Important Notes

- **Never commit your `.env` file** - it contains sensitive API keys
- **The SQLite database** can grow large with many PDFs
- **Local embeddings** require initial download but work offline
- **Groq API** has rate limits on the free tier

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [Phi Data](https://phidata.com/) for the excellent framework
- [Groq](https://groq.com/) for fast AI inference
- [Sentence Transformers](https://www.sbert.net/) for local embeddings# AIpdfassistant
