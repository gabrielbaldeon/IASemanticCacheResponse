using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Sitecore.Configuration;
using Sitecore.Diagnostics;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.IO;
using System.Linq;

namespace IASemanticCacheResponse.SemanticAnalyzer
{
    public class OnnxEmbeddingService : IDisposable
    {
        private readonly InferenceSession _session;
        private readonly Dictionary<string, int> _vocab;
        private readonly int _maxSequenceLength = 256;


        /// <summary>
        /// Loads the ONNX model and vocabulary on service initialization.
        /// </summary>
        public OnnxEmbeddingService()
        {
            try
            {
                // Load model and vocabulary
                string modelPath = Sitecore.IO.FileUtil.MapPath(
                    Settings.GetSetting("AICache.ModelPath"));

                string vocabPath = Sitecore.IO.FileUtil.MapPath(
                    Settings.GetSetting("AICache.VocabPath"));

                _session = new InferenceSession(modelPath);
                _vocab = LoadVocabulary(vocabPath);
            }
            catch (Exception ex)
            {
                Log.Error("Error initializing ONNX service: " + ex.Message, this);
                throw;
            }
        }

        private DenseTensor<long> CreateAttentionMask(long[] tokenIds)
        {
            var attentionMask = new DenseTensor<long>(new long[tokenIds.Length], new[] { 1, tokenIds.Length });
            for (int i = 0; i < tokenIds.Length; i++)
            {
                attentionMask[0, i] = 1;
            }
            return attentionMask;
        }

        /// <summary>
        /// Preprocesses the input text: lowercases, removes stopwords, and performs basic lemmatization.
        /// </summary>
        /// <param name="text">Input text.</param>
        /// <returns>Preprocessed text string.</returns>
        private string PreprocessText(string text)
        {
            // 1. Convert to lowercase
            text = text.ToLowerInvariant();

            // 2. Remove domain-specific stopwords
            var stopWordsSetting = ConfigurationManager.AppSettings["SemanticCache.StopWords"];
            var domainStopWords = stopWordsSetting?.Split(',') ?? new string[0];
            text = string.Join(" ", text.Split(' ')
                .Where(word => !domainStopWords.Contains(word) && word.Length > 2));

            // 3. Basic lemmatization (reduce words to root form)
            text = text.Replace("types", "type")
                       .Replace("exist", "exist")
                       .Replace("products", "product");

            return text;
        }

        /// <summary>
        /// Converts the preprocessed text into a list of token IDs using the vocabulary, including phrase matching.
        /// </summary>
        /// <param name="text">Preprocessed input text.</param>
        /// <returns>List of token IDs.</returns>
        private List<int> AdvancedTokenize(string text)
        {
            // Tokenization that handles compound words and key phrases
            var tokens = new List<int>();
            var words = text.Split(' ');

            // Handle compound phrases
            for (int i = 0; i < words.Length; i++)
            {
                if (i < words.Length - 1)
                {
                    string phrase = $"{words[i]}_{words[i + 1]}";
                    if (_vocab.ContainsKey(phrase))
                    {
                        tokens.Add(_vocab[phrase]);
                        i++; // Skip next word
                        continue;
                    }
                }

                // Handle individual words
                tokens.Add(_vocab.ContainsKey(words[i]) ?
                          _vocab[words[i]] :
                          _vocab["[UNK]"]);
            }

            return tokens.Take(_maxSequenceLength - 1).ToList();
        }

        /// <summary>
        /// Normalizes the raw embedding tensor to a unit vector.
        /// </summary>
        /// <param name="output">ONNX model output tensor.</param>
        /// <returns>Normalized embedding.</returns>
        private float[] NormalizeEmbedding(Tensor<float> output)
        {
            var embedding = new float[output.Dimensions[2]];
            float magnitude = 0.0f;

            // Copy and calculate magnitude
            for (int i = 0; i < embedding.Length; i++)
            {
                embedding[i] = output[0, 0, i];
                magnitude += embedding[i] * embedding[i];
            }

            // Normalize to unit vector
            magnitude = (float)Math.Sqrt(magnitude);
            for (int i = 0; i < embedding.Length; i++)
            {
                embedding[i] = magnitude > 0 ? embedding[i] / magnitude : 0;
            }

            return embedding;
        }

        /// <summary>
        /// Generates a semantic embedding for the input text using the ONNX model.
        /// </summary>
        /// <param name="text">Input text string.</param>
        /// <returns>Normalized embedding vector.</returns>
        public float[] GenerateEmbedding(string text)
        {
            try
            {
                // 1. Advanced text preprocessing
                text = PreprocessText(text);

                // 2. Enhanced tokenization
                var tokens = AdvancedTokenize(text);
                var tokenIds = tokens.Select(t => (long)t).ToArray();

                // 3. Create input tensors
                var inputIds = new DenseTensor<long>(tokenIds, new[] { 1, tokenIds.Length });

                var attentionMask = new DenseTensor<long>(new long[tokenIds.Length], new[] { 1, tokenIds.Length });
                for (int i = 0; i < tokenIds.Length; i++) attentionMask[0, i] = 1;

                var tokenTypeIds = new DenseTensor<long>(new long[tokenIds.Length], new[] { 1, tokenIds.Length });

                // 4. Run the model
                using (var results = _session.Run(new[]
                {
                    NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
                    NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask),
                    NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIds)
                }))
                {
                    var output = results.First().AsTensor<float>();
                    return NormalizeEmbedding(output);
                }
            }
            catch (Exception ex)
            {
                Log.Error($"Error generating embedding: {ex.Message}", this);
                return new float[384];
            }
        }
              
        private Dictionary<string, int> LoadVocabulary(string vocabPath)
        {
            var vocab = new Dictionary<string, int>();
            var lines = File.ReadAllLines(vocabPath);
            for (int i = 0; i < lines.Length; i++)
            {
                vocab[lines[i]] = i;
            }
            return vocab;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
