


import pandas as pd
import json
import os
from collections import defaultdict

# 要忽略的字段
ignore_fields = set([
	'pictureInfo', 'userId', 'orderId',
	'bankCardInfo.bankCardNo', 'bankCardInfo.reservePhoneNo',
	'idInfo.idNumber', 'idInfo.name',
	'linkmanList.0.name', 'linkmanList.0.phone',
	'linkmanList.1.name', 'linkmanList.1.phone',
	'phone',
	# 新增忽略字段
	'area',
	'bankCardInfo.bankName',
	'city',
	'companyInfo.companyAddress',
	'companyInfo.companyName',
	'deviceInfo',
	'email',
	'idInfo.address',
	'idInfo.issuedBy',
	'idInfo.validityDate',
	'liveAddress'
])

def flatten_json(y, prefix='', object_fields=None, ignore_prefixes=None):
	out = {}
	if ignore_prefixes is None:
		ignore_prefixes = set()
	# 如果当前前缀属于被忽略字段或其子字段，直接跳过
	for ignore in ignore_prefixes:
		if prefix.startswith(ignore + '.') or prefix == ignore + '.':
			return out
	if y is None:
		if object_fields and prefix[:-1] in object_fields:
			for subkey in object_fields[prefix[:-1]]:
				out[f'{prefix}{subkey}'] = None
		return out
	if isinstance(y, dict):
		for k, v in y.items():
			out.update(flatten_json(v, f'{prefix}{k}.', object_fields, ignore_prefixes))
	elif isinstance(y, list):
		for i, v in enumerate(y):
			out.update(flatten_json(v, f'{prefix}{i}.', object_fields, ignore_prefixes))
	else:
		out[prefix[:-1]] = y
	return out

def get_all_object_fields(samples):
	object_fields = {}
	for msg in samples:
		if not isinstance(msg, dict):
			continue
		for k, v in msg.items():
			if isinstance(v, dict):
				if k not in object_fields:
					object_fields[k] = set()
				object_fields[k].update(v.keys())
			elif isinstance(v, list):
				for item in v:
					if isinstance(item, dict):
						if k not in object_fields:
							object_fields[k] = set()
						object_fields[k].update(item.keys())
	return object_fields

df = pd.read_csv('data/20250903.csv')

# 收集样本，获取对象属性
msg_samples = []
for _, row in df.iterrows():
	try:
		msg = json.loads(row['报文'])
		msg_samples.append(msg)
	except Exception:
		continue
object_fields = get_all_object_fields(msg_samples)

all_fields = set()
partner_data = defaultdict(list)

for _, row in df.iterrows():
	partner_code = row['合作方编号']
	try:
		msg = json.loads(row['报文'])
	except Exception:
		continue
	# 递归忽略所有以ignore_fields为前缀的字段
	flat = flatten_json(msg, object_fields=object_fields, ignore_prefixes=ignore_fields)
	# 移除所有对象类型的主字段（如bankCardInfo），只保留其属性列
	for obj in object_fields:
		if obj in flat:
			del flat[obj]
	all_fields.update(flat.keys())
	label = 1 if str(row['结果']).strip() == '成功' else 0
	flat['label'] = label
	partner_data[partner_code].append(flat)


# 过滤掉所有被忽略字段及其子字段的表头
def is_not_ignored(field):
	for ignore in ignore_fields:
		if field == ignore or field.startswith(ignore + "."):
			return False
	return True

all_fields = [f for f in sorted(all_fields) if is_not_ignored(f)]
all_fields.append('label')

os.makedirs('partner_csv', exist_ok=True)
sep = '\t'  # 用tab分隔

for partner_code, records in partner_data.items():
	out_path = os.path.join('partner_csv', f'{partner_code}.csv')
	with open(out_path, 'w', encoding='utf-8') as f:
		f.write(sep.join(all_fields) + '\n')
		for rec in records:
			row = [str(rec.get(field, '')) for field in all_fields]
			f.write(sep.join(row) + '\n')