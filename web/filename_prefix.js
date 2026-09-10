import { app } from "../../scripts/app.js";
import { applyTextReplacements } from "../../scripts/utils.js";

app.registerExtension({
    name: "ComfyUI-HPSv3.FilenamePrefix",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "HPSv3PPScore") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated
                ? onNodeCreated.apply(this, arguments)
                : undefined;
            const widget = this.widgets.find((w) => w.name === "filename_prefix");
            widget.serializeValue = () => applyTextReplacements(app, widget.value);
            return result;
        };
    },
});
