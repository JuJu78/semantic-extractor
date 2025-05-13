# Semantic Extractor

## Introduction
This project is a Semantic Extractor that utilizes a Model Context Protocol (MCP) server to process and analyze data. It is designed to be deployed easily and can be integrated with Claude Desktop for enhanced functionality.

## Features
- Semantic data extraction
- Easy deployment
- Integration with Claude Desktop

## Requirements
- Python 3.x
- Virtual environment setup
- Required Python packages (listed in `requirements.txt`)

## Setup and Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/JuJu78/semantic-extractor.git
   cd semantic-extractor
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the MCP Server
To start the MCP server, run the following command:
```bash
python app.py
```

## Deployment

### Deploying Locally
1. Ensure all dependencies are installed.
2. Run the server using the command above.

### Deploying on Claude Desktop
1. Package your application for deployment.
2. Follow the Claude Desktop deployment guide to upload and manage your application.

## Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.
