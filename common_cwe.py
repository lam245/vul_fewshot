import re

def extract_number_from_string(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else None


print(extract_number_from_string("CWE-0234"))  # Output: 0234

sven_cwe_ids = [22, 78, 79, 89, 125, 190, 416, 476, 787]

def use(bigVul_cwe):
    """
    If false, don't use it
    """
    return (extract_number_from_string(bigVul_cwe) in sven_cwe_ids) or bigVul_cwe is None

full_common_cwe = [121, 122, 123, 124, 126, 127, 128, 131, 135, 14, 170, 188, 194, 195, 196, 242, 243, 244, 364, 401, 415, 463, 464, 466, 467, 468, 469, 479, 482, 483, 558, 560, 562, 676, 690, 762, 781, 782, 785, 806, 839, 843]