
# 纽约地铁：站台/方向到站与时刻查询

## 功能说明

当用户询问纽约地铁（NYC Subway、MTA）某一站点、某一方向站台、某条线路的下一班车、预计到站时间或「时刻表」类信息时，按下列步骤用 **fetch_url** 拉取 **JSON** 并整理回复；说明数据来自第三方聚合接口、与 MTA 官方实时源之间存在延迟，不构成官方承诺。

## 术语对应（中文 ↔ NYC）

- **站点**：多线共构的「车站」；API 用 `stationSlug`（如 `72-st-n-q-r`），**不要手猜**，先用搜索接口解析。
- **站台/方向**：纽约地铁常用 **uptown（北向）** / **downtown（南向）** 描述；与 MTA `stop_id` 里的 `N`/`S` 方向一致。若用户说「往曼哈顿」「往布鲁克林」，先根据线路走向映射到 uptown/downtown，不确定时在回复里并列两种方向的数据并注明推断依据。
- **时刻表**：严格意义的**纸质/计划时刻**来自 MTA **GTFS 静态数据**（见文末官方入口）；下列接口侧重 **实时预计到站**（下一班、多班），更接近乘客常用的「到站表」。

## 推荐数据源（JSON，适合 fetch_url）

基础地址：`https://nyc-subway-status.com`（无需 API Key；返回 JSON 文本，非 HTML）。

1. **解析站名 → slug**（必做第一步）  
   使用 `fetch_url` 请求：  
   `https://nyc-subway-status.com/api/search?q=` + **URL 编码**后的关键词（英文站名 + 可选线路，如 `Times+Square+1`、`72+St+Q`）。

2. **某站全部线路的到站**（对应「整个车站各站台方向」）  
   `https://nyc-subway-status.com/api/stops/{stationSlug}`  
   从 JSON 的 `data` 中读取各线路的 **uptown / downtown** 分组与 `minutes_away`、时间戳等字段。

3. **某站某一条线的到站**（对应「某站台/某线」最精确）  
   `https://nyc-subway-status.com/api/stops/{stationSlug}/lines/{routeSlug}`  
   `routeSlug` 为小写：`q`、`b`、`7`、`si`、`gs` 等。

4. **可选：发现端点与字段说明**  
   `https://nyc-subway-status.com/api`  
   纯文本说明（给模型速查）：`https://nyc-subway-status.com/llms.txt`

## 回复用户时的格式建议

- 写清：**车站名**、**线路**、**方向（uptown/downtown）**、**下一班及后续若干班的分钟数或时刻**（按接口给出的字段）。  
- 若 `ok` 为 `false` 或 `NOT_FOUND`：说明未匹配到站点，请用户改用英文站名、或补充线路号后重试搜索。  
- 始终加一句：**非 MTA 官方应用**，以站台显示屏与 [MTA 官网](https://new.mta.info/) 为准。

## 官方与计划时刻（补充）

- MTA 开发者与实时 **GTFS-RT**（Protobuf，不适合直接用 fetch_url 阅读）：<https://api.mta.info/>  
- 计划/静态时刻与数据说明：<https://www.mta.info/developers>  
若用户明确要求「官方公布的计划时刻」而非实时预测，应说明需下载/解析 **GTFS** 或查阅 MTA 公布的 PDF/HTML 时刻表，并可用 `fetch_url` 打开 **new.mta.info** 上对应线路的排班页面（HTML 会转为文本，需自行提取相关段落）。

## 示例流程

1. 用户：「时代广场 1 号线 downtown 下一班几点？」  
2. `fetch_url` → `https://nyc-subway-status.com/api/search?q=Times+Square+42+St+1`  
3. 从结果中取建议的 `stationSlug`，再 `fetch_url` →  
   `https://nyc-subway-status.com/api/stops/{stationSlug}/lines/1`  
4. 提取 **downtown** 侧到站列表，用本地时区友好格式回复。
