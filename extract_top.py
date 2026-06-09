import re


def extract_top_archs(log_file_path):
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r"Score:\s*(-?\d+\.\d+)\s*\|\s*Arch:\s*(\S+)"
    matches = re.findall(pattern, content)

    unique_archs = {}
    for score_str, arch in matches:
        score = float(score_str)
        if arch not in unique_archs or score > unique_archs[arch]:
            unique_archs[arch] = score

    sorted_archs = sorted(unique_archs.items(), key=lambda x: x[1], reverse=True)[:10]

    print("candidate_archs = [")
    for rank, (arch, score) in enumerate(sorted_archs, 1):
        group_name = f"{rank}"
        name_str = f"Score_{score:.4f}"

        print(f'    {{"group": "{group_name}", "name": "{name_str}",')
        print(f'     "arch": "{arch}"}},')
    print("]")


if __name__ == '__main__':
    extract_top_archs('evolution_search.log')