Frequently Asked Questions
==========================

Training with Multiple Nodes
----------------------------

RLite implements distributed training based on Ray. All users need to do is configure the Ray cluster and call ``rlite.init()``.

Specifically, users first need to SSH into the master node and start the Ray cluster:

.. code-block:: bash

    # On master node
    ray start --head
    
Then, users need to SSH into other nodes and start Ray workers:

.. code-block:: bash

    # On other nodes
    ray start --address='<master_node_ip>:6379'

After that, users can start training with the same code:

.. code-block:: bash

    # On master node
    python my_script.py
