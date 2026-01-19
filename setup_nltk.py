#!/usr/bin/env python3
"""
Setup script to download NLTK data
"""
import nltk
import os
import sys

def setup_nltk_data():
    """Download all required NLTK data"""
    print("🧠 Setting up NLTK data...")
    
    # Set NLTK data path
    nltk_data_path = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_path, exist_ok=True)
    nltk.data.path.append(nltk_data_path)
    
    # Required packages
    packages = [
        'punkt',
        'punkt_tab',
        'averaged_perceptron_tagger',
        'maxent_ne_chunker',
        'words',
        'stopwords',
        'wordnet',
        'omw-eng'
    ]
    
    for package in packages:
        try:
            print(f"📦 Downloading {package}...", end=" ")
            nltk.download(package, quiet=False)
            print("✅ Done")
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Verify downloads
    print("\n🔍 Verifying downloads...")
    for package in packages:
        try:
            if package == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif package == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
            elif package == 'averaged_perceptron_tagger':
                nltk.data.find('taggers/averaged_perceptron_tagger')
            elif package == 'maxent_ne_chunker':
                nltk.data.find('chunkers/maxent_ne_chunker')
            elif package == 'words':
                nltk.data.find('corpora/words')
            print(f"✅ {package} verified")
        except Exception as e:
            print(f"❌ {package} not found: {e}")
    
    print("\n✨ NLTK setup complete!")

if __name__ == "__main__":
    setup_nltk_data()