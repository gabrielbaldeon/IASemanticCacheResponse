using Sitecore;
using Sitecore.Caching;
using Sitecore.Configuration;
using System;

namespace IASemanticCacheResponse.SemanticAnalyzer
{
    public class SitecoreAICache : CustomCache
    {
        public SitecoreAICache() : base("SitecoreAICache", StringUtil.ParseSizeString(Settings.GetSetting("AICache.MaxSize")))
        {
        }

        public void Add(string key, object value, TimeSpan expiration)
        {
            DateTime expirationDate = DateTime.Now.Add(expiration);
            SetObject(key, value, expirationDate);
        }

        public object Get(string key)
        {
            return GetObject(key);
        }

        public void Remove(string key)
        {
            base.Remove(key);
        }
    }
}