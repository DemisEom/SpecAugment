"""Image warping using sparse flow defined at control points."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import skimage
from scipy.interpolate import interp2d
from skimage.transform import warp

def _get_grid_locations(image_height, image_width):
  """Wrapper for np.meshgrid."""

  y_range = np.linspace(0, image_height - 1, image_height)
  x_range = np.linspace(0, image_width - 1, image_width)
  y_grid, x_grid = np.meshgrid(y_range, x_range, indexing='ij')
  return np.stack((y_grid, x_grid), -1)


def _expand_to_minibatch(np_array, batch_size):
  """Tile arbitrarily-sized np_array to include new batch dimension."""
  tiles = [batch_size] + [1] * np_array.ndim
  return np.tile(np.expand_dims(np_array, 0), tiles)


def _get_boundary_locations(image_height, image_width, num_points_per_edge):
  """Compute evenly-spaced indices along edge of image."""
  y_range = np.linspace(0, image_height - 1, num_points_per_edge + 2)
  x_range = np.linspace(0, image_width - 1, num_points_per_edge + 2)
  ys, xs = np.meshgrid(y_range, x_range, indexing='ij')
  is_boundary = np.logical_or(
      np.logical_or(xs == 0, xs == image_width - 1),
      np.logical_or(ys == 0, ys == image_height - 1))
  return np.stack([ys[is_boundary], xs[is_boundary]], axis=-1)


def _add_zero_flow_controls_at_boundary(control_point_locations,
                                        control_point_flows, image_height,
                                        image_width, boundary_points_per_edge):

  # batch_size = tensor_shape.dimension_value(control_point_locations.shape[0])
  batch_size = control_point_locations.shape[0]

  boundary_point_locations = _get_boundary_locations(image_height, image_width,
                                                     boundary_points_per_edge)

  boundary_point_flows = np.zeros([boundary_point_locations.shape[0], 2])

  type_to_use = control_point_locations.dtype
  # boundary_point_locations = constant_op.constant(
  #     _expand_to_minibatch(boundary_point_locations, batch_size),
  #     dtype=type_to_use)
  boundary_point_locations = _expand_to_minibatch(boundary_point_locations, batch_size)

  # boundary_point_flows = constant_op.constant(
  #     _expand_to_minibatch(boundary_point_flows, batch_size), dtype=type_to_use)
  boundary_point_flows = _expand_to_minibatch(boundary_point_flows, batch_size)

  # merged_control_point_locations = array_ops.concat(
  #     [control_point_locations, boundary_point_locations], 1)

  merged_control_point_locations = np.concatenate(
      [control_point_locations, boundary_point_locations], 1)

  # merged_control_point_flows = array_ops.concat(
  #     [control_point_flows, boundary_point_flows], 1)

  merged_control_point_flows = np.concatenate(
      [control_point_flows, boundary_point_flows], 1)

  return merged_control_point_locations, merged_control_point_flows


def sparse_image_warp_np(image,
                      source_control_point_locations,
                      dest_control_point_locations,
                      interpolation_order=2,
                      regularization_weight=0.0,
                      num_boundary_points=0):

  # image = ops.convert_to_tensor(image)
  # source_control_point_locations = ops.convert_to_tensor(
  #     source_control_point_locations)
  # dest_control_point_locations = ops.convert_to_tensor(
  #     dest_control_point_locations)

  control_point_flows = (
      dest_control_point_locations - source_control_point_locations)

  clamp_boundaries = num_boundary_points > 0
  boundary_points_per_edge = num_boundary_points - 1

  # batch_size, image_height, image_width, _ = image.get_shape().as_list()
  batch_size, image_height, image_width, _ = list(image.shape)

  # This generates the dense locations where the interpolant
  # will be evaluated.

  grid_locations = _get_grid_locations(image_height, image_width)

  flattened_grid_locations = np.reshape(grid_locations,
                                          [image_height * image_width, 2])

    # flattened_grid_locations = constant_op.constant(
    #     _expand_to_minibatch(flattened_grid_locations, batch_size), image.dtype)

  flattened_grid_locations = _expand_to_minibatch(flattened_grid_locations, batch_size)

  if clamp_boundaries:
    (dest_control_point_locations,
     control_point_flows) = _add_zero_flow_controls_at_boundary(
         dest_control_point_locations, control_point_flows, image_height,
         image_width, boundary_points_per_edge)

    # flattened_flows = interpolate_spline.interpolate_spline(
    #     dest_control_point_locations, control_point_flows,
    #     flattened_grid_locations, interpolation_order, regularization_weight)
  flattened_flows = sp.interpolate.spline(
        dest_control_point_locations, control_point_flows,
        flattened_grid_locations, interpolation_order, regularization_weight)

    # dense_flows = array_ops.reshape(flattened_flows,
    #                                 [batch_size, image_height, image_width, 2])
  dense_flows = np.reshape(flattened_flows,
                                    [batch_size, image_height, image_width, 2])

    # warped_image = dense_image_warp.dense_image_warp(image, dense_flows)
  warped_image = warp(image, dense_flows)

  return warped_image, dense_flows


def dense_image_warp(image, flow):
    # batch_size, height, width, channels = (array_ops.shape(image)[0],
    #                                        array_ops.shape(image)[1],
    #                                        array_ops.shape(image)[2],
    #                                        array_ops.shape(image)[3])
    batch_size, height, width, channels = (np.shape(image)[0],
                                           np.shape(image)[1],
                                           np.shape(image)[2],
                                           np.shape(image)[3])

    # The flow is defined on the image grid. Turn the flow into a list of query
    # points in the grid space.
    # grid_x, grid_y = array_ops.meshgrid(
    #     math_ops.range(width), math_ops.range(height))
    # stacked_grid = math_ops.cast(
    #     array_ops.stack([grid_y, grid_x], axis=2), flow.dtype)
    # batched_grid = array_ops.expand_dims(stacked_grid, axis=0)
    # query_points_on_grid = batched_grid - flow
    # query_points_flattened = array_ops.reshape(query_points_on_grid,
    #                                            [batch_size, height * width, 2])
    grid_x, grid_y = np.meshgrid(
        np.range(width), np.range(height))
    stacked_grid = np.cast(
        np.stack([grid_y, grid_x], axis=2), flow.dtype)
    batched_grid = np.expand_dims(stacked_grid, axis=0)
    query_points_on_grid = batched_grid - flow
    query_points_flattened = np.reshape(query_points_on_grid,
                                        [batch_size, height * width, 2])
    # Compute values at the query points, then reshape the result back to the
    # image grid.
    interpolated = interp2d(image, query_points_flattened)
    interpolated = np.reshape(interpolated,
                              [batch_size, height, width, channels])
    return interpolated

