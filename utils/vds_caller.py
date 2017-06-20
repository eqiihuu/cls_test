# coding=utf-8
"""
Vds feature caller
"""
import json

from utils.cache_caller import CacheCaller

QA_URL = "http://query-analysis/"

cc_qa = CacheCaller(QA_URL)

def call_qa_with_debug(query, domain):
  url = u"{}?query={}&context={}&debug=true".format(QA_URL, query, domain)
  res = cc_qa.http_get(url)
  return json.loads(res)

class Annotation(object):
  def __init__(self, anno_obj):
    self.start_index = anno_obj["start_index"]
    self.end_index = anno_obj["end_index"]
    self.start_token = anno_obj["start_token"]
    self.end_token = anno_obj["end_token"]
    self.raw_str = anno_obj["raw_str"]
    self.tag = anno_obj['value']['tag']
    if "valid_data_type" in anno_obj['value']:
      self.valid_data_type = anno_obj['value']['valid_data_type']

    self.usage = anno_obj['value']['usage']
    self.__type = anno_obj["type"]


def get_annotations(query, domain):
  try:
      qa_json = call_qa_with_debug(query, domain)
      rtn_annotations = []
      for anno_obj in qa_json['debug']['VD_tag_debug_info']['VD_tag_resp']['annotation']:
        anno = Annotation(anno_obj)
        rtn_annotations.append(anno)
      return rtn_annotations
  except Exception as ex:
    print "[ERROR] error to get VDS annotation: " + str(ex)
    return []


# Return the QA result in json format
def get_anno_json(query, domain):
  try:
      qa_json = call_qa_with_debug(query, domain)
      return qa_json
  except Exception as ex:
    print "[ERROR] error to get QA JSON: " + str(ex)
    return []


def match(vd_annotations, start_index, end_index):
  matched_annos = []
  for anno in vd_annotations:
    if anno.start_index <= start_index and \
       anno.end_index >= end_index and \
       anno.usage in ['normal', 'name_entity']:
      matched_annos.append(anno)
  return set([anno.tag for anno in matched_annos])


if __name__ == '__main__':
  query = u"导航去附近的银行"
  domain = "nlu.navigation"
  print get_annotations(query, domain)
