using IASemanticCacheResponse.Models;
using Sitecore.Configuration;
using Sitecore.Diagnostics;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Configuration;
using System.Linq;

namespace IASemanticCacheResponse.SemanticAnalyzer
{
    public class SemanticCacheService : IDisposable
    {
        private readonly SitecoreAICache _cache;
        private readonly OnnxEmbeddingService _embeddingService;
        private readonly ConcurrentDictionary<string, CacheEntry> _cacheIndex;
        private readonly object _lock = new object();

        private double SimilarityThreshold => Double.Parse(Settings.GetSetting("SemanticCache.SimilarityThreshold"));
        private int CacheExpirationHours => Int32.Parse(Settings.GetSetting("SemanticCache.ExpirationHours"));
        private int MaxCacheEntries => Int32.Parse(Settings.GetSetting("SemanticCache.MaxEntries"));

        public SemanticCacheService()
        {
            _cache = new SitecoreAICache();
            _embeddingService = new OnnxEmbeddingService();
            _cacheIndex = new ConcurrentDictionary<string, CacheEntry>();

            // Initialize index from existing cache
            InitializeCacheIndex();
        }

        private void InitializeCacheIndex()
        {
            try
            {
                // 1. Get all cache keys
                var cacheKeys = GetAllCacheKeys();

                // 2. Retrieve entries and load into index
                foreach (var key in cacheKeys)
                {
                    var entry = _cache.Get(key) as CacheEntry;
                    if (entry != null)
                    {
                        _cacheIndex[key] = entry;
                    }
                }

                Log.Info($"[SemanticCache] Cache index initialized with {_cacheIndex.Count} entries", this);
            }
            catch (Exception ex)
            {
                Log.Error("[SemanticCache] Error initializing cache index: " + ex.Message, ex, this);
            }
        }

        private IEnumerable<string> GetAllCacheKeys()
        {
            // Specific implementation to get keys
            // Depends on how SitecoreAICache is implemented
            // This is a placeholder implementation:
            return new List<string>();
        }

        /// <summary>
        /// Retrieves a response for a user query, using cached responses when similar queries exist.
        /// </summary>
        /// <param name="query">User's query string.</param>
        /// <param name="aiService">Function to call the real AI if no match is found.</param>
        /// <returns>Response string.</returns>
        public SysResponse GetResponse(string query, Func<string, string> aiService)
        {
            SysResponse res = new SysResponse();

            // 1. Generate semantic embedding
            float[] embedding = _embeddingService.GenerateEmbedding(query);
            Log.Info($"Generated embedding for: '{query}'", this);

            // 2. Search for semantic match WITH QUERY PARAMETER
            var cacheEntry = FindSimilarCacheEntry(query, embedding);

            if (cacheEntry != null)
            {
                Log.Info($"Cache HIT! Query: '{query}' matches '{cacheEntry.Query}'", this);
                res.similarity = cacheEntry.Similarity;
                res.response = cacheEntry.Response;
                res.source = "Cache";
                return res;
            }

            // 3. If no match, call AI service
            Log.Info($"Cache MISS - New query: '{query}'", this);
            string response = aiService(query);

            res.response= response;
            res.source= "AI";

            // 4. Store in cache
            CacheResponse(query, response, embedding);
            Log.Info($"Response cached for: '{query}'", this);

            return res;
        }

        /// <summary>
        /// Compares the input embedding with cached embeddings to find a similar one.
        /// </summary>
        /// <param name="query">Original user query.</param>
        /// <param name="queryEmbedding">Embedding of the user query.</param>
        /// <returns>Best matching cache entry or null.</returns>
        private CacheEntry FindSimilarCacheEntry(string query, float[] queryEmbedding)
        {
            if (queryEmbedding == null || queryEmbedding.Length == 0)
            {
                Log.Warn("Query embedding is null or empty", this);
                return null;
            }

            CacheEntry bestMatch = null;
            float highestSimilarity = 0;

            foreach (var entry in _cacheIndex.Values)
            {
                if (entry.Embedding == null || entry.Embedding.Length != queryEmbedding.Length)
                {
                    Log.Warn($"Invalid embedding in cache entry: {entry.Query}", this);
                    continue;
                }

                // 1. Compute semantic similarity
                float combinedSimilarity = CalculateCosineSimilarity(queryEmbedding, entry.Embedding);                

                Log.Info($"Similarity for '{entry.Query}': Combined={combinedSimilarity:F2}", this);

                if (combinedSimilarity > highestSimilarity && combinedSimilarity >= SimilarityThreshold)
                {                    
                    highestSimilarity = combinedSimilarity;
                    entry.Similarity = combinedSimilarity.ToString();
                    bestMatch = entry;
                }
            }

            return bestMatch;
        }

        private float CalculateKeywordSimilarity(HashSet<string> keywords1, HashSet<string> keywords2)
        {
            if (keywords1.Count == 0 || keywords2.Count == 0) return 0;

            int intersection = keywords1.Intersect(keywords2).Count();
            int union = keywords1.Union(keywords2).Count();

            return (float)intersection / union;
        }

        /// <summary>
        /// Computes the cosine similarity between two vectors.
        /// </summary>
        private float CalculateCosineSimilarity(float[] vecA, float[] vecB)
        {
            if (vecA == null || vecB == null || vecA.Length != vecB.Length)
                return 0;

            float dot = 0;
            float magA = 0;
            float magB = 0;

            for (int i = 0; i < vecA.Length; i++)
            {
                dot += vecA[i] * vecB[i];
                magA += vecA[i] * vecA[i];
                magB += vecB[i] * vecB[i];
            }

            if (magA <= 0 || magB <= 0)
                return 0;

            return dot / ((float)Math.Sqrt(magA) * (float)Math.Sqrt(magB));
        }

        /// <summary>
        /// Caches a new response and updates the index. Triggers cleanup if needed.
        /// </summary>
        private void CacheResponse(string query, string response, float[] embedding)
        {
            lock (_lock)
            {
                try
                {
                    // 1. Create cache entry
                    var signature = Guid.NewGuid().ToString();
                    var entry = new CacheEntry
                    {
                        Query = query,
                        Response = response,
                        Embedding = embedding,
                        Created = DateTime.Now
                    };

                    // 2. Store in Sitecore cache
                    _cache.Add(signature, entry, TimeSpan.FromHours(CacheExpirationHours));

                    // 3. Store in local index
                    if (_cacheIndex.TryAdd(signature, entry))
                    {
                        Log.Info($"Added to cache index: {signature}", this);
                    }
                    else
                    {
                        Log.Warn($"Failed to add to cache index: {signature}", this);
                    }

                    // 4. Cache cleanup
                    CleanupCache();
                }
                catch (Exception ex)
                {
                    Log.Error("Error caching response: " + ex.Message, ex, this);
                }
            }
        }

        private void CleanupCache()
        {
            try
            {
                // 1. Remove expired entries
                var expiredKeys = _cacheIndex
                    .Where(e => (DateTime.Now - e.Value.Created).TotalHours > CacheExpirationHours)
                    .Select(e => e.Key)
                    .ToList();

                foreach (var key in expiredKeys)
                {
                    if (_cacheIndex.TryRemove(key, out _))
                    {
                        _cache.Remove(key);
                    }
                }

                // 2. Limit maximum size
                while (_cacheIndex.Count > MaxCacheEntries)
                {
                    var oldest = _cacheIndex.OrderBy(e => e.Value.Created).First();
                    if (_cacheIndex.TryRemove(oldest.Key, out _))
                    {
                        _cache.Remove(oldest.Key);
                    }
                }
            }
            catch (Exception ex)
            {
                Log.Error("Error cleaning cache: " + ex.Message, ex, this);
            }
        }

        public void Dispose()
        {
            _embeddingService?.Dispose();
        }

        public class CacheEntry
        {
            public string Query { get; set; }
            public string Response { get; set; }
            public float[] Embedding { get; set; }
            public DateTime Created { get; set; }
            public string Similarity { get; set; }
        }
    }
}
