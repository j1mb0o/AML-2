import torch
import torch.nn as nn

from networks import Conv4

class MAML(nn.Module):

    def __init__(self, num_ways, input_size, T=1, second_order=False, inner_lr=0.4, **kwargs):
        super().__init__()
        self.num_ways = num_ways
        self.input_size = input_size
        self.num_updates = T
        self.second_order = second_order
        self.inner_loss = nn.CrossEntropyLoss()
        self.inner_lr = inner_lr

        self.network = Conv4(self.num_ways, img_size=int(input_size**0.5)) 


    # controller input = image + label_previous
    def apply(self, x_supp, y_supp, x_query, y_query, training=False):
            """
            Pefrosmt the inner-level learning procedure of MAML: adapt to the given task 
            using the support set. It returns the predictions on the query set, as well as the loss
            on the query set (cross-entropy).
            You may want to set the gradients manually for the base-learner parameters 

            :param x_supp (torch.Tensor): the support input iamges of shape (num_support_examples, num channels, img width, img height)
            :param y_supp (torch.Tensor): the support ground-truth labels
            :param x_query (torch.Tensor): the query inputs images of shape (num_query_inputs, num channels, img width, img height)
            :param y_query (torch.Tensor): the query ground-truth labels

            :returns:
            - query_preds (torch.Tensor): the predictions of the query inputs
            - query_loss (torch.Tensor): the cross-entropy loss on the query inputs
            """
            init_w = [param.clone() for param in self.network.parameters()]

            for _ in range(self.num_updates):
                fast_weights = [param.clone() for param in init_w]

                support_set_predictions = self.network(x_supp, weights=fast_weights)
                support_set_loss = self.inner_loss(support_set_predictions, y_supp)
                gradients = torch.autograd.grad(support_set_loss, fast_weights, create_graph=self.second_order)

                fast_weights= [weight - self.inner_lr * grad for weight, grad in zip(fast_weights, gradients)]

            query_preds = self.network(x_query, weights=fast_weights)
            query_loss = self.inner_loss(query_preds, y_query)

            if training:
                query_loss.backward()

            return query_preds, query_loss