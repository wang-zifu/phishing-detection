import re
import sys
import traceback
from ipaddress import ip_address
from urllib.parse import parse_qsl, urlparse

import numpy as np
import pandas as pd
import tld


def compute_entropy(labels, base=np.e):
    _, counts = np.unique(labels, return_counts=True)
    norm_counts = counts / counts.sum()
    return -(norm_counts * np.log(norm_counts) / np.log(base)).sum()

def compute_ratio(a, b):
    if b == 0:
        return a
    return compute_ratio(b, a % b)

def count_digit(s):
    return sum(c.isdigit() for c in s)

def count_alpha(s):
    return sum(c.isalpha() for c in s)

def count_alnum(s):
    return sum(c.isalnum() for c in s)

def count_sp_char(s):
    sp_chars = "$-_.!*'(),;/?:@=&"
    return sum(c in sp_chars for c in s)


class Extractor:
    def __init__(self, url):
        self._url = urlparse(url)
        self._host = self._url.port
        self._dirs = re.subn('/{2,}', '/', self._url.path)[0].split('/')[:-1]
        self._name = self._url.path.rsplit('/', 1)[-1]
        self._dom = None if self.domain_is_ip() else tld.get_tld(url, as_object=True)

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
        return compute_ratio(self.url_n_digit(), self.url_n_alpha())

    def url_rate_digit(self) -> float:
        return self.url_n_digit() / self.url_len()

    def url_rate_character_continuity(self) -> float:
        # TODO
        pass

    def tld_in_suspicious_list(self) -> bool:
        # TODO
        pass

    def domain_is_ip(self) -> bool:
        try:
            ip_address(self._url.hostname)
        except ValueError:
            return False
        return True

    def domain_len(self) -> int:
        return len(self._url.hostname)

    def domain_n_digit(self) -> int:
        return sum(c.isdigit() for c in self._url.hostname)

    def domain_n_nonalnum(self) -> int:
        return sum(not c.isalnum() for c in self._url.hostname)

    def domain_n_hyphen(self) -> int:
        return self._url.hostname.count('-')

    def domain_n_at_sign(self) -> int:
        return self._url.hostname.count('@')

    def domain_entropy(self) -> float:
        return compute_entropy(list(self._url.hostname))

    def primary_domain_len(self) -> int:
        fld = self._url.hostname if self._dom is None else self._dom.fld
        return len(fld)

    def primary_domain_n_digit(self) -> int:
        fld = self._url.hostname if self._dom is None else self._dom.fld
        return sum(c.isdigit() for c in fld)

    def primary_domain_n_nonalnum(self) -> int:
        fld = self._url.hostname if self._dom is None else self._dom.fld
        return sum(not c.isalnum() for c in fld)

    def primary_domain_n_hyphen(self) -> int:
        fld = self._url.hostname if self._dom is None else self._dom.fld
        return fld.count('-')

    def primary_domain_n_at_sign(self) -> int:
        fld = self._url.hostname if self._dom is None else self._dom.fld
        return fld.count('@')

    def primary_domain_in_alexa_top_100(self) -> bool:
        # TODO
        pass

    def primary_domain_entropy(self) -> float:
        fld = self._url.hostname if self._dom is None else self._dom.fld
        return compute_entropy(list(fld))

    def subdomain_len(self) -> int:
        if self._dom is None:
            return 0
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
        return np.mean([len(dir_) for dir_ in self._dirs])

    def path_maxlen_dir(self) -> int:
        if not self._dirs:
            return 0
        return max(len(dir_) for dir_ in self._dirs)

    def path_dir_rate_digit(self) -> float:
        joined = '/'.join(self._dirs)
        return sum(c.isdigit() for c in joined)

    def path_n_double_slash(self) -> int:
        return self._url.path.count('//')

    def path_n_zero(self) -> int:
        return self._url.path.count('0')

    def path_n_digit(self) -> int:
        return sum(c.isdigit() for c in self._url.path)

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
        return compute_ratio(uppers, lowers)

    def path_rate_digit(self) -> float:
        path_len = self.path_len()
        if path_len == 0:
            return 0
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
        return compute_ratio(self.query_len(), self.path_len())

    def ratio_query_url(self) -> float:
        return compute_ratio(self.query_len(), self.url_len())

    def ratio_query_domain(self) -> float:
        return compute_ratio(self.query_len(), self.domain_len())

    def ratio_path_url(self) -> float:
        return compute_ratio(self.path_len(), self.url_len())

    def ratio_path_domain(self) -> float:
        return compute_ratio(self.path_len(), self.domain_len())

    def ratio_domain_url(self) -> float:
        return compute_ratio(self.domain_len(), self.url_len())

    def name_len(self) -> int:
        return len(self._name)

    def name_n_digit(self) -> int:
        return sum(c.isdigit() for c in self._name)

    def name_rate_digit(self) -> float:
        name_len = self.name_len()
        if name_len == 0:
            return 0
        return self.name_n_digit() / name_len


def feature_extract(urls, csv):
    features = []
    try:
        for url in urls:
            # print(url)
            try:
                e = Extractor(url)
            except tld.exceptions.TldBadUrl:
                continue
            except tld.exceptions.TldDomainNotFound:
                continue
            features.append([
                e.url_len(),
                e.url_n_alpha(),
                e.url_n_ampersand(),
                e.url_n_digit(),
                e.url_n_dot(),
                e.url_n_equal(),
                e.url_n_question_mark(),
                e.url_n_semicolon(),
                e.url_n_sp_char(),
                e.url_n_underscore(),
                e.url_rate_digit(),
                e.url_ratio_digit_letter(),
                e.domain_is_ip(),
                e.domain_len(),
                e.domain_n_at_sign(),
                e.domain_n_digit(),
                e.domain_n_hyphen(),
                e.domain_n_nonalnum(),
                e.primary_domain_entropy(),
                e.primary_domain_len(),
                e.primary_domain_n_at_sign(),
                e.primary_domain_n_digit(),
                e.primary_domain_n_hyphen(),
                e.primary_domain_n_nonalnum(),
                e.subdomain_len(),
                e.subdomain_n(),
                e.subdomain_n_dot(),
                e.path_avglen_dir(),
                e.path_dir_rate_digit(),
                e.path_len(),
                e.path_maxlen_dir(),
                e.path_n_digit(),
                e.path_n_dir(),
                e.path_n_double_slash(),
                e.path_n_sp_char(),
                e.path_n_zero(),
                e.path_percent20_in(),
                e.path_rate_digit(),
                e.path_ratio_upper_lower(),
                e.path_single_char_dir_in(),
                e.path_upper_dir_in(),
                e.params_len(),
                e.query_len(),
                e.query_n(),
                e.query_n_digit(),
                e.name_len(),
                e.name_n_digit(),
                e.name_rate_digit(),
                e.ratio_domain_url(),
                e.ratio_path_domain(),
                e.ratio_path_url(),
                e.ratio_query_domain(),
                e.ratio_query_path(),
                e.ratio_query_url(),
            ])
    except Exception as e:
        exc_info = sys.exc_info()
        print(url)
        traceback.print_exception(*exc_info)
        return
    
    features = pd.DataFrame(features)
    features.to_csv(csv)


if __name__ == '__main__':
    urls = open('20190312-github.com-ebubekirbbr-phishing_url_detection-phishtank_3-url.txt').readlines()
    feature_extract(urls)
