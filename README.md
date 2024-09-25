## Natural Image Search (NIS)

This is a simple image search application that uses the CLIP model from OpenAI to find images based on a textual description. 

**Features:**

* **Semantic Search:** Search for images using natural language descriptions instead of keywords.
* **Fast Indexing:**  Uses pre-computed image embeddings for faster search results.
* **Caching:**  Saves computed embeddings to disk to speed up future searches.
* **Dark Theme:** Features a clean and modern dark theme.

**How It Works:**

1. **Image Indexing:** The application first indexes all the images in a selected directory by computing their image embeddings using the CLIP model. These embeddings capture the semantic meaning of the images.
2. **Text Embedding:** When you enter a search query, the app computes the text embedding of your query using CLIP.
3. **Similarity Search:** The app compares the text embedding to the image embeddings and returns the images that are most similar to your query.

**Installation:**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/natural-image-search.git
   cd natural-image-search 
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

**Usage:**

1. **Run the App:**
   ```bash
   python app.py
   ```

2. **Select Image Directory:** Click the "Select the desired folder" button to choose the folder containing the images you want to search. 

3. **Search:**  Type a description of the image you're looking for in the search bar. The app will display the most relevant images in the grid below.

4. **View Full Image:** Click on a thumbnail to view the full-resolution image.

**Example Search Queries:**

* "A photo of a dog playing in a park"
* "A painting of a sunset over the ocean"
* "A picture of a city street at night" 

**Requirements:**

* Python 3.7 or later
* PyTorch 
* transformers
* PyQt6
* Pillow (PIL) 

**Contributing:**

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements. 

**License:**

This project is licensed under the MIT License. 
