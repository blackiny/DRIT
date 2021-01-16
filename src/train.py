import torch
from options import TrainOptions
from dataset import dataset_unpair
from model import DRIT
from saver import Saver

def main():
  cuda = True if torch.cuda.is_available() else False
  # parse options
  parser = TrainOptions()
  opts = parser.parse()

  # daita loader
  print('\n--- load dataset ---')
  dataset = dataset_unpair(opts)
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

  # model
  print('\n--- load model ---')
  model = DRIT(opts)
  model.setgpu(opts.gpu)
  if opts.resume is None:
    model.initialize()
    ep0 = -1
    total_it = 0
  else:
    ep0, total_it = model.resume(opts.resume)
  model.set_scheduler(opts, last_ep=ep0)
  ep0 += 1
  print('start the training at epoch %d'%(ep0))

  # saver for display and output
  saver = Saver(opts)
  # if graph displayed
  graph_display = False
  # train
  print('\n--- train ---')
  max_it = 500000
  for ep in range(ep0, opts.n_ep):
    for it, (images_a, images_b) in enumerate(train_loader):
      if images_a.size(0) != opts.batch_size or images_b.size(0) != opts.batch_size:
        continue

      # input data
      images_a = images_a.cuda(opts.gpu).detach()
      images_b = images_b.cuda(opts.gpu).detach()

      # update model
      if (it + 1) % opts.d_iter != 0 and it < len(train_loader) - 2:
        model.update_D_content(images_a, images_b)
        continue
      else:
        model.update_D(images_a, images_b)
        model.update_EG()

      # save to display graph
      if not graph_display:
        saver.write_model_display(model = model, input_images=(images_a, images_b))
        graph_display = True
      # save to display file
      saver.write_display(total_it, model, not opts.no_display_img, True)

      print('total_it: %d (ep %d, it %d), lr %08f' % (total_it, ep, it, model.gen_opt.param_groups[0]['lr']))
      total_it += 1
      if total_it >= max_it:
        saver.write_img(-1, model)
        saver.write_model(-1, total_it, model)
        break

    # decay learning rate
    if opts.n_ep_decay > -1:
      model.update_lr()

    # save result image
    saver.write_img(ep, model)

    # Save network weights
    saver.write_model(ep, total_it, model)

  return

if __name__ == '__main__':
  main()
