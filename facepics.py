import wikipedia
import json
import time
import os

# Identify the bot for Wikipedia
wikipedia.set_user_agent("GreenlightCinemaBot/1.0 (advait.m@example.com)")

def fetch_actor_images():
    # 1. Load the actors list
    try:
        with open("actors_list.txt", "r", encoding="utf-8") as f:
            actors = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("Error: actors_list.txt not found!")
        return

    # 2. Load existing progress if it exists to avoid starting over
    output_file = "face.constraint"
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            face_constraints = json.load(f)
        print(f"Resuming from existing file. Already found {len(face_constraints)} actors.")
    else:
        face_constraints = {}

    print(f"Total list size: {len(actors)} actors.")

    # 3. Process the list
    for i, actor in enumerate(actors):
        if actor in face_constraints:
            continue  # Skip already fetched actors
            
        print(f"[{i+1}/{len(actors)}] Searching: {actor}")
        
        try:
            time.sleep(2.0) # Respect rate limits
            
            # Use search and then page
            search_results = wikipedia.search(actor)
            if not search_results:
                continue
                
            # Fetch page, auto_suggest=False prevents weird redirects
            page = wikipedia.page(search_results[0], auto_suggest=False)
            
            # Logic: Try to find a valid image, fallback to first if none
            if page.images:
                valid = [img for img in page.images if img.endswith(('.jpg', '.jpeg', '.png'))]
                face_constraints[actor] = valid[0] if valid else page.images[0]
            
            # Save incrementally so you don't lose progress on a crash
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(face_constraints, f, indent=4)
                
        except Exception as e:
            print(f" Skipping {actor}: {type(e).__name__}")
            continue

    print(f"\n Finished! Processed {len(face_constraints)} total actors.")

if __name__ == "__main__":
    fetch_actor_images()