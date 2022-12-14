def construct_graph(graph_dict, inputs, outputs):
    queue = inputs[:]
    make_dict = {}
    for key, val in graph_dict.items():
        if key in inputs:
            make_dict[key] = tf.placeholder(tf.float32, name=key)
        else:
            make_dict[key] = None
    # Breadth-First search of graph starting from inputs
    while len(queue) != 0:
        cur = graph_dict[queue[0]]
        for outg in cur["outgoing"]:
            if make_dict[outg[0]]: # If discovered node, do add/multiply operation
                make_dict[outg[0]] = tf.add(make_dict[outg[0]], tf.multiply(outg[1], make_dict[queue[0]]))
            else: # If undiscovered node, input is just coming in multiplied and add outgoing to queue
                make_dict[outg[0]] = tf.multiply(make_dict[queue[0]], outg[1])
                for outgo in graph_dict[outg[0]]["outgoing"]:
                    queue.append(outgo[0])
        queue.pop(0)
    # Returns one data graph for each output
    return [make_dict[x] for x in outputs]
