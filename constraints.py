import duckdb
import json

def generate_modern_constraints(db_path="movies_analytics.db", output_path="market_constraints.json"):
    con = duckdb.connect(db_path)
    print("Extracting Modern Market Constraints from DuckDB (Continuous Seasonal Format)...")

    payload = {
        "metadata": {
            "source": "DuckDB Analytics Engine - Modern Era V7",
            "timeframe": "2015 - Present",
            "min_budget": "$500,000",
            "filters": "Live-Action, English Language, Min 500 Votes"
        }
    }
    # 1. Top Genres (Highest Revenue to Budget Ratio)
    # DYNAMIC FETCH: Get every unique genre from the database, ignoring "Unknown"
    genre_query = """
    SELECT DISTINCT TRIM(UNNEST(string_split(genre_str, ','))) AS genre_name
    FROM movie_analytics
    WHERE genre_str IS NOT NULL AND genre_str != ''
    """
 
    genres_to_test = [row[0] for row in con.execute(genre_query).fetchall() if row[0] and row[0].lower() != 'unknown']
    genre_stats = [] #creation of empty set
    
    for genre in genres_to_test:
        query = f"""
        SELECT ROUND(AVG(revenue / budget), 2) as avg_roi
        FROM movie_analytics
        WHERE budget >= 500000 AND revenue > 0 AND release_year >= 2015
          AND genre_str LIKE '%{genre}%'
        HAVING COUNT(*) >= 10;
        """
        result = con.execute(query).fetchone()
        if result and result[0]:
            genre_stats.append({"genre": genre, "revenue_to_budget_ratio": result[0]})
            
    payload["top_genres"] = sorted(genre_stats, key=lambda x: x['revenue_to_budget_ratio'], reverse=True)[:15]

    # 2. Seasonal Fit (Continuous Quarterly ROI for Top Genres)
    print("Mapping Quarterly Trends for Top Genres...")
    
    # Grab the names of the top 5 genres we just calculated
    top_genre_names = [g["genre"] for g in payload["top_genres"]] #takes data from new generate list of top genre
    
    seasonal_fit_dict = {}
    
    for genre in top_genre_names:
        seasonal_query = f"""
        SELECT 
            CASE 
                WHEN release_month BETWEEN 1 AND 3 THEN 'Q1'
                WHEN release_month BETWEEN 4 AND 6 THEN 'Q2'
                WHEN release_month BETWEEN 7 AND 9 THEN 'Q3'
                ELSE 'Q4'
            END AS quarter,
            ROUND(AVG(revenue / budget), 2) as avg_roi
        FROM movie_analytics
        WHERE budget >= 500000 AND release_year >= 2015 AND release_month IS NOT NULL
          AND genre_str LIKE '%{genre}%'
        GROUP BY quarter
        ORDER BY quarter;
        """
        results = con.execute(seasonal_query).fetchall()
        # Create a dictionary of Q1-Q4 ROI for this specific genre
        seasonal_fit_dict[genre] = {row[0]: row[1] for row in results}

    payload["seasonal_fit"] = seasonal_fit_dict

    # 3. Directors Impact
    director_query = """
    SELECT director, ROUND(AVG(revenue / budget), 2) as avg_roi
    FROM movie_analytics
    WHERE budget >= 500000 AND release_year >= 2015 AND director != 'Unknown'
      AND original_language = 'en' AND genre_str NOT LIKE '%Animation%' AND vote_count >= 500
    GROUP BY director
    HAVING COUNT(*) >= 3
    ORDER BY avg_roi DESC
    LIMIT 10;
    """
    payload["director_impact"] = [f"{row[0]} ({row[1]}x ROI)" for row in con.execute(director_query).fetchall()]

    # 4. Top Actors (Mainstream)
    top_actor_query = """
    WITH split_cast AS (
        SELECT TRIM(UNNEST(string_split(cast_leads, ','))) AS actor_name, revenue, budget
        FROM movie_analytics
        WHERE budget >= 500000 AND release_year >= 2005 
          AND cast_leads != 'Unknown' AND cast_leads IS NOT NULL
          AND original_language = 'en' AND genre_str NOT LIKE '%Animation%' AND vote_count >= 500
    )
    SELECT actor_name, ROUND(AVG(revenue / budget), 2) as avg_roi
    FROM split_cast
    WHERE actor_name != ''
    GROUP BY actor_name
    HAVING COUNT(*) >= 5
    ORDER BY avg_roi DESC
    LIMIT 10;
    """
    payload["top_actors"] = [f"{row[0]} ({row[1]}x ROI)" for row in con.execute(top_actor_query).fetchall()]

    # 5. Emerging Actors (Mainstream)
    actor_query = """
    WITH split_cast AS (
        SELECT TRIM(UNNEST(string_split(cast_leads, ','))) AS actor_name, revenue, budget
        FROM movie_analytics
        WHERE budget >= 500000 AND release_year >= 2022
          AND cast_leads != 'Unknown' AND cast_leads IS NOT NULL
          AND original_language = 'en' AND vote_count >= 500
    )
    SELECT actor_name, ROUND(AVG(revenue / budget), 2) as avg_roi
    FROM split_cast
    WHERE actor_name != ''
    GROUP BY actor_name
    HAVING COUNT(*) BETWEEN 2 AND 4
    ORDER BY avg_roi DESC
    LIMIT 10;
    """
    payload["emerging_actors"] = [f"{row[0]} ({row[1]}x ROI)" for row in con.execute(actor_query).fetchall()]

    # 6. Top 3 Actors by Genre (The new block)
    print("Calculating top 3 actors per genre...")
    top_actors_by_genre = {}
    
    for genre in top_genre_names:
        genre_actor_query = f"""
        WITH split_cast AS (
            SELECT TRIM(UNNEST(string_split(cast_leads, ','))) AS actor_name, revenue, budget
            FROM movie_analytics
            WHERE budget >= 500000 AND release_year >= 2015
              AND cast_leads != 'Unknown' AND cast_leads IS NOT NULL
              AND original_language = 'en' AND genre_str LIKE '%{genre}%'
              AND vote_count >= 500
        )
        SELECT actor_name, ROUND(AVG(revenue / budget), 2) AS avg_roi
        FROM split_cast
        WHERE actor_name != ''
        GROUP BY actor_name
        HAVING COUNT(*) >= 3
        ORDER BY avg_roi DESC
        LIMIT 3;
        """
        results = con.execute(genre_actor_query).fetchall()
        top_actors_by_genre[genre] = [f"{row[0]} ({row[1]}x ROI)" for row in results]

    payload["top_actors_by_genre"] = top_actors_by_genre

    # Write out the clean JSON file
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=4)
        
    print(f"Success! Continuous seasonal formatting written to {output_path}")
    con.close()

# Ensure the function handles the connection safely
if __name__ == "__main__":
    try:
        generate_modern_constraints()
    except Exception as e:
        print(f"An error occurred: {e}")