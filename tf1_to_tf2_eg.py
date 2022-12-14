import tensorflow as tf

def construct_graph(graph_dict, inputs, outputs):
    queue = inputs[:]
    make_dict = {}
    for key, val in graph_dict.items():
        if key in inputs:
            # Use keras.Input instead of placeholders
            make_dict[key] = tf.keras.Input(name=key, shape=(), dtype=tf.dtypes.float32)
        else:
            make_dict[key] = None
    # Breadth-First search of graph starting from inputs
    while len(queue) != 0:
        cur = graph_dict[queue[0]]
        for outg in cur["outgoing"]:
            if make_dict[outg[0]] is not None: # If discovered node, do add/multiply operation
                make_dict[outg[0]] = tf.keras.layers.add([
                    make_dict[outg[0]],
                    tf.keras.layers.multiply(
                        [[outg[1]], make_dict[queue[0]]],
                    )],
                )
            else: # If undiscovered node, input is just coming in multiplied and add outgoing to queue
                make_dict[outg[0]] = tf.keras.layers.multiply(
                    [make_dict[queue[0]], [outg[1]]]
                )
                for outgo in graph_dict[outg[0]]["outgoing"]:
                    queue.append(outgo[0])
        queue.pop(0)
    # Returns one data graph for each output
    model_inputs = [make_dict[key] for key in inputs]
    model_outputs = [make_dict[key] for key in outputs]
    return [tf.keras.Model(inputs=model_inputs, outputs=o) for o in model_outputs]

def main():
    graph_def = {
        "B": {
            "incoming": [],
            "outgoing": [("A", 1.0)]
        },
        "C": {
            "incoming": [],
            "outgoing": [("A", 1.0)]
        },
        "A": {
            "incoming": [("B", 2.0), ("C", -1.0)],
            "outgoing": [("D", 3.0)]
        },
        "D": {
            "incoming": [("A", 2.0)],
            "outgoing": []
        }
    }
    outputs = construct_graph(graph_def, ["B", "C"], ["A"])
    print("Builded models:", outputs)
    for o in outputs:
        o.summary(120)
        print("Output:", o((1.0, 1.0)))

if __name__ == "__main__":
    main()
