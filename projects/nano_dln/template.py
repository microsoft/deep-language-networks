from dataclasses import dataclass
from typing import List
import os
import glob
import yaml
import logging
from jinja2 import Template
from packaging import version as pkg_version


@dataclass
class DLNTemplate:
    template: str
    stop_tokens: List[str] = None
    version: int = "latest"
    description: str = None
    message: str = None
    message_alternatives: List[str] = None

    def render(self, **kwargs):
        if kwargs.get("message") is None:
            kwargs["message"] = self.message

        return Template(self.template).render(**kwargs).lstrip().rstrip()


class Templates:
    _instance = None

    def __init__(self):
        self._data = {}
        template_directory = os.path.join(os.path.dirname(__file__), 'templates/')
        for filename in glob.glob(f"{template_directory}/*.yaml"):
            template_name = os.path.basename(filename).split(".")[0]
            template = yaml.safe_load(open(filename, "r"))

            self._data[template_name] = []
            for tversion, ttemplate in template.items():
                if "v" not in tversion:
                    raise ValueError("Version must be in the format v1, v1.2, etc.")

                ttemplate["version"] = pkg_version.parse(tversion.split("v")[-1])
                if "stop_tokens" in ttemplate:
                    # strip the first \ of \\n from the stop tokens
                    for i, stop_token in enumerate(ttemplate["stop_tokens"]):
                        ttemplate["stop_tokens"][i] = ttemplate["stop_tokens"][
                            i
                        ].replace("\\n", "\n")
                self._data[template_name].append(DLNTemplate(**ttemplate))

    @staticmethod
    def get(template_name):
        template_name, _, version = template_name.partition(":")
        if not version:
            version = "latest"

        if Templates._instance is None:
            Templates._instance = Templates()

        templates = Templates._instance._data[template_name]

        if version == "latest":
            template = max(templates, key=lambda x: x.version)
        else:
            template = [
                t for t in templates if t.version == pkg_version.parse(version)
            ][0]

        logging.info(f"Loaded template {template_name} v{template.version}")
        return template


def load_template(template_name):
    return Templates.get(template_name)
