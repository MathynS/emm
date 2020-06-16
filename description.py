from typing import Union


class Description:

    def __init__(self, attribute: str, value: Union[str, float, int, bool] = None):
        if attribute == 'all':
            value = 'all'
        self.description = dict()
        self.description[attribute] = value

    def __contains__(self, col):
        return col in self.description

    def extend(self, attribute, value):
        if 'all' in self.description:
            self.description = dict()
            self.description[attribute] = value
        else:
            self.description[attribute] = value
        return self

    def decrypt(self, translation):
        for key, value in self.description.items():
            if key in translation:
                self.description[key] = translation[key][value]

    def __str__(self):
        if 'all' in self.description:
            return 'all'
        else:
            result, result2 = [], []
            for key, value in self.description.items():
                if isinstance(value, list):
                    result.append(f"{value[0]} < {key} <= {value[1]}")
                else:
                    result.append(f"{key} = {value}")
            length = 0
            for i, item in enumerate(result):
                length += len(item)
                if length > 30:
                    result2.append("<br>")
                    length = 0
                result2.append(item)
            return " AND ".join(result2)
