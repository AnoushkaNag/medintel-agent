def clean_region(region):

    if not isinstance(region, str):
        return "Unknown"

    region = region.lower()

    region = region.replace(" region", "")
    region = region.replace(" municipality", "")
    region = region.replace(" district", "")

    region = region.strip()

    return region.title()