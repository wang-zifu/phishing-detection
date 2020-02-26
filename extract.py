import re
from ipaddress import ip_address
from urllib.parse import parse_qsl, urlparse

import numpy as np
from tld import get_tld


def compute_entropy(labels, base=np.e):
    _, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()

def count_digit(s):
    return sum(c.isdigit() for c in s)

def count_alpha(s):
    return sum(c.alpha() for c in s)

def count_alnum(s):
    return sum(c.alnum() for c in s)

def count_sp_char(s):
    sp_chars = "$-_.!*'(),;/?:@=&"
    return sum(c in sp_chars for c in s)


class Extractor:
    def __init__(self, url):
        self._url = urlparse(url)
        self._dirs = re.subn('/{2,}', '/', self._url.path).split('/')[:-1]
        self._name = self._url.path.rsplit('/', 1)[-1]
        self._dom = None if self.domain_is_ip() else get_tld(url)

    def url_len(self) -> int:
        return len(self._url.geturl())

    def url_n_semicolon(self) -> int:
        return self._url.geturl().count(';')

    def url_n_underscore(self) -> int:
        return self._url.geturl().count('_')

    def url_n_question_mark(self) -> int:
        return self._url.geturl().count('?')

    def url_n_equal(self) -> int:
        return self._url.geturl().count('=')

    def url_n_ampersand(self) -> int:
        return self._url.geturl().count('&')

    def url_n_dot(self) -> int:
        return self._url.geturl().count('.')

    def url_n_digit(self) -> int:
        return count_digit(self._url.geturl())

    def url_n_alpha(self) -> int:
        return count_alpha(self._url.geturl())

    def url_n_sp_char(self) -> int:
        return count_sp_char(self._url.geturl())

    def url_ratio_digit_letter(self) -> float:
        return self.url_n_digit() / self.url_n_alpha()

    def url_rate_digit(self) -> float:
        return self.url_n_digit() / self.url_len()

    def tld_in_suspicious_list(self) -> bool:
        # TODO
        pass

    def domain_is_ip(self) -> bool:
        try:
            ip_address(self._url.netloc)
        except ValueError:
            return False
        return True

    def domain_len(self) -> int:
        return len(self._url.netloc)

    def domain_n_digit(self) -> int:
        return sum(c.isdigit() for c in self._url.netloc)

    def domain_n_nonalnum(self) -> int:
        return sum(not c.isalnum() for c in self._url.netloc)

    def domain_n_hyphen(self) -> int:
        return self._url.netloc.count('-')

    def domain_n_at_sign(self) -> int:
        return self._url.netloc.count('@')

    def _domain_entropy(self) -> float:
        return compute_entropy(list(self._url.netloc))

    def primary_domain_len(self) -> int:
        return len(self._dom.fld)

    def primary_domain_n_digit(self) -> int:
        return sum(c.isdigit() for c in self._dom.fld)

    def primary_domain_n_nonalnum(self) -> int:
        return sum(not c.isalnum() for c in self._dom.fld)

    def primary_domain_n_hyphen(self) -> int:
        return self._dom.fld.count('-')

    def primary_domain_n_at_sign(self) -> int:
        return self._dom.fld.count('@')

    def primary_domain_in_alexa_top_100(self) -> bool:
        # TODO
        pass

    def primary_domain_entropy(self) -> float:
        return compute_entropy(list(self._dom.fld))

    def subdomain_len(self) -> int:
        return len(self._dom.subdomain)

    def subdomain_n(self) -> int:
        if self._dom is None:
            return 0
        return len(self._dom.subdomain.split('.'))

    def subdomain_n_dot(self) -> int:
        if self._dom is None:
            return 0
        return self._dom.subdomain.count('.')

    def path_len(self) -> int:
        return len(self._url.path)

    def path_n_dir(self) -> int:
        return len(self._dirs)

    def path_avglen_dir(self) -> float:
        return np.mean(len(dir_) for dir_ in self._dirs)
    
    def path_maxlen_dir(self) -> int:
        return max(len(dir_) for dir_ in self._dirs)
    
    def path_dir_rate_digit(self) -> float:
        joined = '/'.join(self._dirs)
        return sum(c.isdigit() for c in joined)

    def path_n_double_slash(self) -> int:
        return self._url.path.count('//')

    def path_n_zero(self) -> int:
        return self._url.path.count('0')

    def path_n_digit(self) -> int:
        return sum(c.isdigit() for c in self._url.path())

    def path_n_sp_char(self) -> int:
        return count_sp_char(self._url.path)

    def path_percent20_in(self) -> bool:
        return '%20' in self._url.path

    def path_upper_dir_in(self) -> bool:
        return any(dir_.isupper() for dir_ in self._dirs)

    def path_single_char_dir_in(self) -> bool:
        return any(len(dir_) == 1 for dir_ in self._dirs)

    def path_ratio_upper_lower(self) -> float:
        uppers = sum(c.isupper() for c in self._url.path)
        lowers = sum(c.islower() for c in self._url.path)
        return uppers / lowers

    def path_rate_digit(self) -> float:
        return self.path_n_digit() / self.path_len()

    def params_len(self) -> int:
        return len(self._url.params)

    def query_len(self) -> int:
        return len(self._url.query)

    def query_n(self) -> int:
        return len(parse_qsl(self._url.query))
    
    def query_n_digit(self) -> int:
        return sum(c.isdigit() for c in self._url.query)

    def ratio_query_path(self) -> float:
        return self.query_len() / self.path_len()

    def ratio_query_url(self) -> float:
        return self.query_len() / self.url_len()

    def ratio_query_domain(self) -> float:
        return self.query_len() / self.domain_len()

    def ratio_path_url(self) -> float:
        return self.path_len() / self.url_len()

    def ratio_path_domain(self) -> float:
        return self.path_len() / self.domain_len()

    def ratio_domain_url(self) -> float:
        return self.domain_len() / self.url_len()

    def rate_character_continuity(self) -> float:
        # TODO
        pass

    def name_len(self) -> int:
        return len(self._name)

    def name_n_digit(self) -> int:
        return sum(c.isdigit() for c in self._name)

    def name_rate_digit(self) -> float:
        return self.name_n_digit() / self.name_len()


if __name__ == '__main__':
    e = Extractor('https://koyo.kr/post/pwnable-kr-unlink/')
    assert(e.url_len() == 39)
