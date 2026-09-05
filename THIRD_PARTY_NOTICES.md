# Third-party software and assets

RoadLabelOps original project code is Copyright (C) 2026 by 未来。
It is licensed under the GNU Affero General Public License v3.0 only
(`AGPL-3.0-only`), as stated in `LICENSE`. That project license does not replace,
remove, or alter the notices and license terms that apply to the third-party
software and assets listed below.

RoadLabelOps uses third-party software under its respective license terms. This
file is a practical inventory, not a replacement for those terms. Exact resolved
versions are recorded in `uv.lock` and `frontend/package-lock.json`.

## Material runtime dependencies

| Component | License declared upstream | Upstream |
| --- | --- | --- |
| CVAT SDK | MIT | <https://github.com/cvat-ai/cvat> |
| FastAPI | MIT | <https://github.com/fastapi/fastapi> |
| HTTPX | BSD-3-Clause | <https://github.com/encode/httpx> |
| Pydantic | MIT | <https://github.com/pydantic/pydantic> |
| Pydantic Settings | MIT | <https://github.com/pydantic/pydantic-settings> |
| python-multipart | Apache-2.0 | <https://github.com/Kludex/python-multipart> |
| PyYAML | MIT | <https://github.com/yaml/pyyaml> |
| Uvicorn | BSD-3-Clause | <https://github.com/Kludex/uvicorn> |
| Ultralytics | AGPL-3.0 | <https://github.com/ultralytics/ultralytics> |
| Ultralytics Platform | AGPL-3.0-only | <https://github.com/ultralytics/sdk> |
| Ultralytics THOP | AGPL-3.0 | <https://github.com/ultralytics/thop> |
| Outfit font package | OFL-1.1 | <https://fontsource.org/fonts/outfit> |
| Phosphor Icons | MIT | <https://github.com/phosphor-icons/react> |
| Next.js | MIT | <https://github.com/vercel/next.js> |
| React / React DOM | MIT | <https://github.com/facebook/react> |

Ultralytics is used through its Python API and is part of the default detection
installation. Review its AGPL terms before redistributing a combined application
or offering a hosted service.

The resolved frontend graph includes `caniuse-lite` (CC-BY-4.0) and may include
Sharp (Apache-2.0) and platform-specific libvips packages
(LGPL-3.0-or-later). Python transitive dependencies include additional
permissive and reciprocal licenses.
Consult the lockfiles and the license files installed with each package when
preparing a binary distribution.

## External programs

RoadLabelOps invokes FFmpeg/ffprobe and connects to a separately operated CVAT
service. They are not included in this source tree. A distributor that bundles
either program must comply with the license terms of the exact build it ships.

## Models, videos and datasets

This source repository does not include model weight files, downloaded videos,
generated datasets, CVAT exports, or release payloads. The project license does
not grant rights to those separate assets. Users may independently obtain Pexels,
Open Images, or other material; users and distributors must verify the applicable
asset terms themselves.

## Packaged releases

Before publishing a wheel, standalone frontend, container image, desktop bundle,
or other prebuilt artifact, generate an inventory from that exact artifact and
include all license and attribution files required by its resolved dependencies.
