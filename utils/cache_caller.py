# coding=utf-8
import codecs
import os
import requests
from urlparse import urlparse


__author__ = 'Xiao Jiang'
__date__ = '17/3/12'
__email__ = 'xjiang@mobvoi.com'


class CacheCaller(object):
    """ cache caller; all result are string with no \n!
    """

    def __init__(self, url):
        """ init a CacheCaller by url. The url is only used for
            cache file name only
        """
        self.url = url
        self.fn_cache = CacheCaller.__get_cache_file_name(url)
        self.__cache_dict = self.__load_cache()
        self.__fw_cache = codecs.open(self.fn_cache, 'a', 'utf8')
        self.__fw_write_count = 0
        self.__http_get_count = 0

    @staticmethod
    def __get_cache_file_name(url):
        up = urlparse(url)
        home_dir = os.path.expanduser("~")
        fn_host = up.netloc
        fn_path = up.path.replace('/', '_')
        fn = ".urlcache_{}{}".format(fn_host, fn_path)
        return os.path.join(home_dir, fn)

    @staticmethod
    def __get_key(url):
        """ get cache key
        """
        return str(hash(url))

    def __do_get(self, url):
        self.__http_get_count += 1
        print u"[http get] {}".format(url)
        return requests.get(url).content.decode('utf8')

    def http_get(self, url):
        """ do HTTP GET, return a string (in utf8 encoding)
        """
        key = CacheCaller.__get_key(url)
        val = None
        if key in self.__cache_dict:
            val = self.__cache_dict[key]
        else:
            val = self.__do_get(url)
            self.__cache_dict[key] = val
            self.__fw_cache.write(u"{}\t{}\n".format(key, val))
            self.__fw_cache.flush()
            self.__fw_write_count += 1
        return val

    def close_cache_writer(self):
        self.__fw_cache.close()
        print u"[cache write] wrote {} cache entries this time".format(self.__fw_write_count)
        print u"[http get] actual http get count: {}".format(self.__http_get_count)

    def __load_cache(self):
        if not os.path.exists(self.fn_cache):
            return {}
        cache_dict = {}
        with codecs.open(self.fn_cache, 'r', 'utf8') as fr:
            for line in fr.xreadlines():
                try:
                    line = line.decode('utf8').strip()
                    if not line or line.startswith('#'):
                        continue
                    key, val = line.split('\t')
                    cache_dict[key] = val
                except Exception:
                    continue
        print u"[load cache] load {} cache entries from {}".format(
                len(cache_dict), self.fn_cache)
        return cache_dict

