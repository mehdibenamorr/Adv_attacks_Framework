import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from utils.differential_evolution import differential_evolution


# Adversarial attacks methods :
"""FGSM Attack"""
def fgsm(Net ,x ,y_true ,epsilon=0.1):
    # Generate Adv Image
    outputs = Net(x)
    loss = Net.SoftmaxWithXent(outputs, y_true)
    loss.backward()  # to obtain gradients of x
    # Add small perturbation
    x_grad = torch.sign(x.grad.data)
    x_adversarial = torch.clamp(x.data + epsilon * x_grad, 0, 1)

    return x_adversarial

"""One pixel Attack"""
def perturb_image(xs,img):
    if xs.ndim < 2:
        xs = np.array([xs])
    batch = len(xs)
    imgs= img.repeat(batch,1,1,1)
    xs= xs.astype(int)

    count =0
    for x in xs:
        pixels = np.split(x,len(x)/3)

        for pixel in pixels:
            x_pos, y_pos, i = pixel
            imgs[count,0, x_pos, y_pos] = (i/255.0-0.1307)/0.3081
        count += 1
    return imgs

def predict_classes(xs, img, target_class, net, cuda, model,minimize=True):
    imgs_perturbed = perturb_image(xs, img.clone())

    with torch.no_grad():
        input = Variable(imgs_perturbed).cuda() if cuda else Variable(imgs_perturbed)
    if model != "CNN":
        input.resize_(input.size(0),28*28)
    predictions = F.softmax(net(input), dim=1).data.cpu().numpy()[:, target_class] if cuda else \
        F.softmax(net(input), dim=1).data.numpy()[:, target_class]

    return predictions if minimize else 1 - predictions

def attack_success(x, img, target_class, net, cuda,model,targeted_attack=False):
    adv_image = perturb_image(x, img.clone())

    with torch.no_grad():
        input = Variable(adv_image).cuda() if cuda else Variable(adv_image)

    if model != "CNN":
        input.resize_(input.size(0), 28 * 28)
    predicted_class = np.argmax(F.softmax(net(input), dim=1).data.cpu().numpy()[0]) if cuda else \
        np.argmax(F.softmax(net(input), dim=1).data.numpy()[0])

    if (targeted_attack and predicted_class == target_class) or (not targeted_attack and predicted_class != target_class):
        return True

def one_pixel(img, label, net, model,target=None, pixels=1, maxiter=75, popsize=400 ,cuda=True):
    # img : 1*784 tensor ==> 1*1*28*28 tensor

    img.resize_(1,1,28,28)

    targeted_attack = target is not None
    target_class = target if targeted_attack else label

    bounds = [(0 ,28), (0 ,28), (0 ,255) ] * pixels

    popmul = int(max(1 , popsize/len(bounds)))

    predict_fn = lambda xs: predict_classes(xs ,img ,target_class ,net ,cuda,model,target is None)

    callback_fn = lambda x, convergence: attack_success(x ,img ,target_class ,net ,model ,targeted_attack)

    inits = np.zeros([popmul *len(bounds), len(bounds)])
    for init in inits:
        for i in range(pixels):
            init[i * 3 + 0 ]= np.random.random() * 28
            init[i * 3 + 1] = np.random.random() * 28
            init[i * 3 + 2] = np.random.normal(128 ,127)

    attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter, popsize=popsize, recombination=1, atol=-1,
                                           callback=callback_fn, polish=False, init=inits)

    adv_img = perturb_image(attack_result.x, img)
    with torch.no_grad():
        adv_var = Variable(adv_img).cuda() if cuda else Variable(adv_img)
    if model!="CNN":
        adv_var.resize_(adv_var.size(0),28*28)

    predicted_probs = F.softmax(net(adv_var) ,dim=1)

    predicted_class = np.argmax(predicted_probs.cpu().data.numpy()) if cuda else np.argmax(predicted_probs.data.numpy())

    if (not targeted_attack and predicted_class!= label) or (targeted_attack and predicted_class == target_class):
        return 1, attack_result.x.astype(int) , adv_var
    return 0, [None] , None


"""L-BFGS Attack"""
def l_bfgs(self ,_x ,_l_target ,norm ,max_iter):

    # Optimitzation box contrained
    for i in range(max_iter):
        self.Optimizer.zero_grad()
        output = self(_x)
        loss = self.SoftmaxWithXent(output, _l_target)

        # Norm used
        if norm == "l1":
            adv_loss = loss + torch.mean(torch.abs(self.r))
        elif norm == "l2":
            adv_loss = loss + torch.mean(torch.pow(self.r, 2))
        else:
            adv_loss = loss

        adv_loss.backward()
        self.Optimizer.step()

        # Until output == y_target
        y_pred_adversarial = np.argmax(self(_x).cpu().data.numpy()) if self.args.cuda else np.argmax(
            self(_x).data.numpy())

        if y_pred_adversarial == _l_target.data.item():
            break

        if i == max_iter - 1:
            print("Results may be incorrect, Optimization run for {} iteration".format(max_iter))
    x_adversarial = _x + self.r

    return x_adversarial, y_pred_adversarial


methods ={'FGSM':fgsm,'L_BFGS':l_bfgs}