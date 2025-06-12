# PDF Search Web Application

A web interface for searching through political party documents using different embedding models.

## Features

- Select from all available database/embedding models (shows all supported models)
- Automatic background database creation when needed
- Real-time status updates for database creation progress
- Enter search queries in a text box
- View full search results with:
  - Party name
  - Filename and chunk ID
  - Similarity score
  - Model used
  - Full content of matching text
- Responsive design that works on desktop and mobile
- Non-blocking UI - webpage remains responsive during database creation

## Setup

1. Install Flask dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure your main `pdf_processor.py` and databases are in the parent directory.

3. Run the application:
```bash
python app.py
```

4. Open your browser to `http://localhost:5000`

## Usage

1. Select a database/model from the dropdown (shows all supported models)
2. If the selected model doesn't have a database yet, it will show a status indicator
3. Enter your search query in the text box
4. Click "Search" or press Enter
5. If a database needs to be created, it will start automatically in the background
6. The page will show status updates and remain responsive during database creation
7. View the results below once the search completes
8. You can change the model and enter new queries without refreshing the page

## Model Status Indicators

- ✓ Available - Database is ready for searching
- ⏳ Creating... - Database is being created in the background
- ✗ Failed - Database creation failed
- (Not created) - Database needs to be created (will happen automatically when you search)

## Requirements

- Flask 2.3.3+
- Your existing PDF processor system with databases
- Modern web browser with JavaScript enabled

## File Structure

```
web-app/
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css     # Styling
│   └── js/
│       └── main.js       # JavaScript functionality
└── templates/
    └── index.html        # Main HTML template
```
