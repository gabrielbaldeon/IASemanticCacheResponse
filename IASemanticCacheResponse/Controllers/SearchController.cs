using IASemanticCacheResponse.Models;
using IASemanticCacheResponse.SemanticAnalyzer;
using System;
using System.Web.Http;


namespace IASemanticCacheResponse.Controllers
{
    public class SearchController : ApiController
    {
        private static readonly SemanticCacheService _cacheService = new SemanticCacheService();

        /// <summary>
        /// Handles incoming user queries and returns a response, using cache when possible.
        /// </summary>
        /// <param name="request">Object containing the user's message.</param>
        /// <returns>AI response or cached response.</returns>
        [HttpPost]
        [Route("api/Search/GetResponse")]
        public IHttpActionResult GetResponse([FromBody] MessageRequest request)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(request?.userMessage))
                {
                    return BadRequest("La consulta no puede estar vacía");
                }

                var data = _cacheService.GetResponse(
                request.userMessage,
                query => CallRealAIService(query)
            );

                return Ok(new
                {
                    success = true,
                    data.response,
                    data.source,
                    data.similarity
                });
            }
            catch (Exception ex)
            {
                Sitecore.Diagnostics.Log.Error($"Error en GetResponse: {ex.Message}\n{ex.StackTrace}", this);
                return InternalServerError(ex);
            }
        }

        /// <summary>
        /// /// <summary>
        /// Simulates an AI service call. Replace with actual AI integration (e.g., OpenAI, Azure).
        /// </summary>
        /// <param name="query">The user's question.</param>
        /// <returns>Simulated AI response.</returns>
        /// </summary>
        /// <param name="query"></param>
        /// <returns></returns>
        private string CallRealAIService(string query)
        {
            // Example - Replace with actual connection to OpenAI, Azure Cognitive Services, etc.

            // Simulate processing
            System.Threading.Thread.Sleep(1000);

            // Simulated response
            return $"For '{query}', the AI recommends: contacting support at '{DateTime.Now.ToShortDateString() + DateTime.Now.ToShortTimeString()}'";
        }
    }
}