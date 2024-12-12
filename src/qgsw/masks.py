"""Staggered Grid Masks."""

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
import torch.nn.functional as F  # noqa: N812


class Masks:
    """Masks for staggered grid used in Shallow-water discretization.

    The variables are:
     - w: vorticity
     - h: layer thickness
     - u: x-axis velocity
     - v: y-axis velocity

        y
        ^

        :           :
        w-----v-----w..
        |           |
        |           |
        u     h     u
        |           |
        |           |
        w-----v-----w..   > x

    """

    def __init__(self, mask_hgrid: torch.Tensor) -> None:
        """Computes automatically masks given the binary mask on the h grid.

        as well as the irregulare boundary points
        Example of mask:
            Given the following 4x3 domain:

                ^

                x----o---- ---- ----
                |    |XXXX|XXXX|XXXX|
                |    |XXXX|XXXX|XXXX|
                x----o----o----o----x
                |    |    |    |    |
                |    |    |    |    |
                x----o---- ---- ----x
                |XXXX|    |    |    |
                |XXXX|    |    |    |
                 ----x----x----x----x   >

            The mask_h would be:
                mask_h = torch.ones(4,3)
                mask_h[0,0] = 0
                mask_h[1:,2] = 0

            Amoung boundary points (x and o) , x are regular boundary points,
            i.e. on the rectangle boundary,
            and o are irregular boundary points.


        Parameters
            - mask_hgrid: float/double Tensor, shape (nx, ny), binary values
        """
        self._mtype = mask_hgrid.dtype
        self._mshape = mask_hgrid.shape

        self._generate_masks(mask_h=mask_hgrid)

        self._compute_vorticity_bounds()

        # Irregular boundary indices
        self._compute_psi_irregular_bounds()

        self._generate_stencils()

        # convert masks to correct data type
        self._convert_types()

    def _make_h_mask(self, mask_h: torch.Tensor) -> torch.Tensor:
        """Create h mask.

        Args:
            mask_h (torch.Tensor): H mask tensor.

        Returns:
            torch.Tensor: H boolean mask.
        """
        new_shape = (1,) * (4 - len(self._mshape)) + self._mshape
        return mask_h.reshape(new_shape)

    def _make_omega_mask(self) -> torch.Tensor:
        """Create vorticity mask.

        Returns:
            torch.Tensor: ω boolean mask.
        """
        h_pool = F.avg_pool2d(self.h, (2, 2), stride=(1, 1), padding=(1, 1))
        return h_pool > 1 / 8

    def _make_u_mask(self) -> torch.Tensor:
        """Create u mask.

        Returns:
            torch.Tensor: U boolean mask.
        """
        h_pool = F.avg_pool2d(self.h, (2, 1), stride=(1, 1), padding=(1, 0))
        return h_pool > 3 / 4

    def _make_v_mask(self) -> torch.Tensor:
        """Create v mask.

        Returns:
            torch.Tensor: V boolean mask.
        """
        h_pool = F.avg_pool2d(self.h, (1, 2), stride=(1, 1), padding=(0, 1))
        return h_pool > 3 / 4

    def _make_psi_mask(self) -> torch.Tensor:
        """Create stream function mask.

        Returns:
            torch.Tensor: Ψ boolean mask
        """
        h_pool = F.avg_pool2d(self.h, (2, 2), stride=(1, 1), padding=(1, 1))
        return h_pool > 7 / 8

    def _generate_masks(self, mask_h: torch.Tensor) -> None:
        """Generate all masks.

        Args:
            mask_h (torch.Tensor): H mask grid.
        """
        # H masks
        self.h = self._make_h_mask(mask_h)
        self.not_h = torch.logical_not(self.h)
        # U masks
        self.u = self._make_u_mask()
        self.not_u = torch.logical_not(self.u)
        # V Masks
        self.v = self._make_v_mask()
        self.not_v = torch.logical_not(self.v)
        # ω Masks
        self.w = self._make_omega_mask()
        self.not_w = torch.logical_not(self.w)
        # Ψ Masks
        self.psi = self._make_psi_mask()
        self.not_psi = torch.logical_not(self.psi)

    def _compute_vorticity_bounds(self) -> None:
        """Compute vorticity boundaries."""
        # Select V points with one neighbor
        typed_v = self.v.type(self._mtype)
        v_neigh_nb = F.avg_pool2d(
            typed_v,
            (2, 1),
            stride=(1, 1),
            padding=(1, 0),
            divisor_override=1,
        )
        v_has_1_neigh = v_neigh_nb == 1
        v_has_not_1_neigh = torch.logical_not(v_has_1_neigh)

        # Select U points with one neighbor
        typed_u = self.u.type(self._mtype)
        u_neigh_nb = F.avg_pool2d(
            typed_u,
            (1, 2),
            stride=(1, 1),
            padding=(0, 1),
            divisor_override=1,
        )
        u_has_1_neigh = u_neigh_nb == 1
        u_has_not_1_neigh = torch.logical_not(u_has_1_neigh)

        # Vorticity Vertical Bounds
        v_has_1_neigh_but_not_u = torch.logical_and(
            v_has_1_neigh,
            u_has_not_1_neigh,
        )
        self.w_vertical_bound = self.w * v_has_1_neigh_but_not_u

        # Vorticity Horizontal Bounds
        u_has_1_neigh_but_not_v = torch.logical_and(
            u_has_1_neigh,
            v_has_not_1_neigh,
        )
        self.w_horizontal_bound = self.w * u_has_1_neigh_but_not_v

        # Vorticity Cornerout
        u_and_v_have_1_neigh = torch.logical_and(u_has_1_neigh, v_has_1_neigh)
        self.w_cornerout_bound = self.w * u_and_v_have_1_neigh

        # Valid Vorticity

        not_vertical_bound = torch.logical_not(self.w_vertical_bound)
        not_horizontal_bound = torch.logical_not(self.w_horizontal_bound)
        not_boundary = torch.logical_and(
            not_vertical_bound,
            not_horizontal_bound,
        )
        not_corner = torch.logical_not(self.w_cornerout_bound)
        not_bound = torch.logical_and(not_boundary, not_corner)

        self.w_valid = self.w * not_bound

    def _compute_psi_irregular_bounds(self) -> None:
        """Compute Psi's irregular bounds."""
        psi_pool = F.avg_pool2d(
            self.psi.type(self._mtype),
            (3, 3),
            stride=(1, 1),
        )

        self.psi_irrbound_xids, self.psi_irrbound_yids = torch.where(
            torch.logical_and(
                self.not_psi[0, 0, 1:-1, 1:-1],
                psi_pool[0, 0] > 1 / 18,
            ),
        )

    def _generate_h_stencil_in_x(self) -> None:
        """Generate h stencil in x direction."""
        self.u_sten_hx_eq2 = torch.logical_and(
            torch.logical_and(
                F.avg_pool2d(
                    self.h.type(self._mtype),
                    (2, 1),
                    stride=(1, 1),
                    padding=(1, 0),
                )
                > 3 / 4,
                F.avg_pool2d(
                    self.h.type(self._mtype),
                    (4, 1),
                    stride=(1, 1),
                    padding=(2, 0),
                )
                < 7 / 8,
            ),
            self.u,
        )
        self.u_sten_hx_gt4 = torch.logical_and(
            torch.logical_not(self.u_sten_hx_eq2),
            self.u,
        )
        self.u_sten_hx_eq4 = torch.logical_and(
            F.avg_pool2d(
                self.h.type(self._mtype),
                (6, 1),
                stride=(1, 1),
                padding=(3, 0),
            )
            < 11 / 12,
            self.u_sten_hx_gt4,
        )
        self.u_sten_hx_gt6 = torch.logical_and(
            torch.logical_not(self.u_sten_hx_eq4),
            self.u_sten_hx_gt4,
        )

    def _generate_h_stencil_in_y(self) -> None:
        """Generate h stencil in y direction."""
        self.v_sten_hy_eq2 = torch.logical_and(
            torch.logical_and(
                F.avg_pool2d(
                    self.h.type(self._mtype),
                    (1, 2),
                    stride=(1, 1),
                    padding=(0, 1),
                )
                > 3 / 4,
                F.avg_pool2d(
                    self.h.type(self._mtype),
                    (1, 4),
                    stride=(1, 1),
                    padding=(0, 2),
                )
                < 7 / 8,
            ),
            self.v,
        )
        self.v_sten_hy_gt4 = torch.logical_and(
            torch.logical_not(self.v_sten_hy_eq2),
            self.v,
        )
        self.v_sten_hy_eq4 = torch.logical_and(
            F.avg_pool2d(
                self.h.type(self._mtype),
                (1, 6),
                stride=(1, 1),
                padding=(0, 3),
            )
            < 11 / 12,
            self.v_sten_hy_gt4,
        )
        self.v_sten_hy_gt6 = torch.logical_and(
            torch.logical_not(self.v_sten_hy_eq4),
            self.v_sten_hy_gt4,
        )

    def _generate_w_stencil_in_x(self) -> None:
        """Generate w stencil in x direction."""
        self.v_sten_wx_eq2 = torch.logical_and(
            torch.logical_and(
                F.avg_pool2d(
                    self.w.type(self._mtype),
                    (2, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                )
                > 3 / 4,
                F.avg_pool2d(
                    self.w.type(self._mtype),
                    (4, 1),
                    stride=(1, 1),
                    padding=(1, 0),
                )
                < 7 / 8,
            ),
            self.v,
        )
        self.v_sten_wx_gt4 = torch.logical_and(
            torch.logical_not(self.v_sten_wx_eq2),
            self.v,
        )
        self.v_sten_wx_eq4 = torch.logical_and(
            F.avg_pool2d(
                self.w.type(self._mtype),
                (6, 1),
                stride=(1, 1),
                padding=(2, 0),
            )
            < 11 / 12,
            self.v_sten_wx_gt4,
        )
        self.v_sten_wx_gt6 = torch.logical_and(
            torch.logical_not(self.v_sten_wx_eq4),
            self.v_sten_wx_gt4,
        )

    def _generate_w_stencil_in_y(self) -> None:
        """Generate w stencil in y direction."""
        self.u_sten_wy_eq2 = torch.logical_and(
            torch.logical_and(
                F.avg_pool2d(
                    self.w.type(self._mtype),
                    (1, 2),
                    stride=(1, 1),
                    padding=(0, 0),
                )
                > 3 / 4,
                F.avg_pool2d(
                    self.w.type(self._mtype),
                    (1, 4),
                    stride=(1, 1),
                    padding=(0, 1),
                )
                < 7 / 8,
            ),
            self.u,
        )
        self.u_sten_wy_gt4 = torch.logical_and(
            torch.logical_not(self.u_sten_wy_eq2),
            self.u,
        )
        self.u_sten_wy_eq4 = torch.logical_and(
            F.avg_pool2d(
                self.w.type(self._mtype),
                (1, 6),
                stride=(1, 1),
                padding=(0, 2),
            )
            < 11 / 12,
            self.u_sten_wy_gt4,
        )
        self.u_sten_wy_gt6 = torch.logical_and(
            torch.logical_not(self.u_sten_wy_eq4),
            self.u_sten_wy_gt4,
        )

    def _generate_stencils(self) -> None:
        """Generate Stencils."""
        self._generate_h_stencil_in_x()
        self._generate_h_stencil_in_y()
        self._generate_w_stencil_in_x()
        self._generate_w_stencil_in_y()

    def _convert_types(self) -> None:
        """Convert masks to correct data type."""
        self.h = self.h.type(self._mtype)
        self.u = self.u.type(self._mtype)
        self.v = self.v.type(self._mtype)
        self.w = self.w.type(self._mtype)
        self.w_vertical_bound = self.w_vertical_bound.type(self._mtype)
        self.w_horizontal_bound = self.w_horizontal_bound.type(self._mtype)
        self.w_valid = self.w_valid.type(self._mtype)
        self.psi = self.psi.type(self._mtype)
        self.not_h = self.not_h.type(self._mtype)
        self.not_u = self.not_u.type(self._mtype)
        self.not_v = self.not_v.type(self._mtype)
        self.not_w = self.not_w.type(self._mtype)
        self.not_psi = self.not_psi.type(self._mtype)

        self.u_sten_hx_eq2 = self.u_sten_hx_eq2.type(self._mtype)
        self.u_sten_hx_eq4 = self.u_sten_hx_eq4.type(self._mtype)
        self.u_sten_hx_gt4 = self.u_sten_hx_gt4.type(self._mtype)
        self.u_sten_hx_gt6 = self.u_sten_hx_gt6.type(self._mtype)

        self.u_sten_wy_eq2 = self.u_sten_wy_eq2.type(self._mtype)
        self.u_sten_wy_eq4 = self.u_sten_wy_eq4.type(self._mtype)
        self.u_sten_wy_gt4 = self.u_sten_wy_gt4.type(self._mtype)
        self.u_sten_wy_gt6 = self.u_sten_wy_gt6.type(self._mtype)

        self.v_sten_hy_eq2 = self.v_sten_hy_eq2.type(self._mtype)
        self.v_sten_hy_eq4 = self.v_sten_hy_eq4.type(self._mtype)
        self.v_sten_hy_gt4 = self.v_sten_hy_gt4.type(self._mtype)
        self.v_sten_hy_gt6 = self.v_sten_hy_gt6.type(self._mtype)

        self.v_sten_wx_eq2 = self.v_sten_wx_eq2.type(self._mtype)
        self.v_sten_wx_eq4 = self.v_sten_wx_eq4.type(self._mtype)
        self.v_sten_wx_gt4 = self.v_sten_wx_gt4.type(self._mtype)
        self.v_sten_wx_gt6 = self.v_sten_wx_gt6.type(self._mtype)

    @classmethod
    def empty(
        cls,
        nx: int,
        ny: int,
        device: torch.device,
    ) -> Self:
        """Create an empty mask.

        Args:
            nx (int): Number of points in the X direction.
            ny (int): Number of points in the Y direction.
            device (torch.device): Device.

        Returns:
            Self: Mask
        """
        return cls(torch.ones((nx, ny), device=device))


if __name__ == "__main__":
    import matplotlib as mpl
    import numpy as np

    mpl.rcParams.update({"font.size": 24})
    import matplotlib.pyplot as plt

    n = 6
    mask_1 = torch.ones(n, n)

    mask_2 = torch.ones(n, n)
    mask_2[1, 0] = 0.0
    mask_2[n - 1, 2] = 0.0
    mask_2[0, n - 2] = 0.0
    mask_2[1, n - 2] = 0.0
    mask_2[0, n - 1] = 0.0
    mask_2[1, n - 1] = 0.0
    mask_2[2, n - 1] = 0.0

    plt.ion()
    for mask, title in [
        (mask_1, "rect. domain"),
        (mask_2, "non-rect. domain"),
    ]:
        masks = Masks(mask)

        for stencil_var in ["h", "w"]:
            f, a = plt.subplots(figsize=(14, 9))
            f.suptitle(f"{title}, {stencil_var}-stencil masks")
            a.imshow(
                mask.T,
                origin="lower",
                cmap="Greys_r",
                interpolation=None,
                vmin=-1,
            )
            (
                a.set_xticks(np.arange(-0.5, n + 0.5)),
                a.set_yticks(np.arange(-0.5, n + 0.5)),
            )
            a.grid()

            # h mask
            size = 90
            size2 = 130
            q_xmin, q_ymin = 0, 0
            mask_h_ids = torch.argwhere(masks.h.squeeze())
            a.scatter(
                q_xmin + mask_h_ids[:, 0],
                q_ymin + mask_h_ids[:, 1],
                s=size,
                marker="o",
                label="h",
                color="mediumseagreen",
            )

            # w, psi plain and boundary mask
            w_xmin, w_ymin = -0.5, -0.5
            # w
            mask_w_ids = torch.argwhere(masks.w.squeeze())
            a.scatter(
                w_xmin + mask_w_ids[:, 0],
                w_ymin + mask_w_ids[:, 1],
                s=size,
                marker="s",
                label="$w$",
                color="pink",
            )
            mask_w_hb_ids = torch.argwhere(masks.w_horizontal_bound.squeeze())
            a.scatter(
                w_xmin + mask_w_hb_ids[:, 0],
                w_ymin + mask_w_hb_ids[:, 1],
                s=2 * size,
                marker="4",
                label="$w$ horiz",
                color="red",
            )
            mask_w_vb_ids = torch.argwhere(masks.w_vertical_bound.squeeze())
            a.scatter(
                w_xmin + mask_w_vb_ids[:, 0],
                w_ymin + mask_w_vb_ids[:, 1],
                s=2 * size,
                marker="2",
                label="$w$ vert",
                color="green",
            )
            mask_w_cob_ids = torch.argwhere(masks.w_cornerout_bound.squeeze())
            a.scatter(
                w_xmin + mask_w_cob_ids[:, 0],
                w_ymin + mask_w_cob_ids[:, 1],
                s=2 * size,
                marker="x",
                label="$w$ c_out",
                color="cyan",
            )
            mask_w_valid_ids = torch.argwhere(masks.w_valid.squeeze())
            a.scatter(
                w_xmin + mask_w_valid_ids[:, 0],
                w_ymin + mask_w_valid_ids[:, 1],
                s=2 * size,
                marker="*",
                label="$w$ valid",
                color="cyan",
                alpha=0.4,
            )
            # psi
            mask_psi_ids = torch.argwhere(masks.psi.squeeze())
            a.scatter(
                w_xmin + mask_psi_ids[:, 0],
                w_ymin + mask_psi_ids[:, 1],
                s=size / 2,
                marker="D",
                label="$\\psi$",
                color="brown",
                alpha=0.5,
            )
            a.scatter(
                w_xmin + 1 + masks.psi_irrbound_xids,
                w_ymin + 1 + masks.psi_irrbound_yids,
                s=size,
                marker="o",
                label="$\\mathcal{I}$",
                color="purple",
                alpha=0.5,
            )

            if stencil_var == "h":
                u_xmin, u_ymin = -0.5, 0
                mask_u_ids = torch.argwhere(masks.u_sten_hx_eq2.squeeze())
                a.scatter(
                    u_xmin + mask_u_ids[:, 0],
                    u_ymin + mask_u_ids[:, 1],
                    s=size2,
                    marker=">",
                    label="u_sten_hx_eq2",
                    color="lightblue",
                )
                mask_u_ids = torch.argwhere(masks.u_sten_hx_eq4.squeeze())
                a.scatter(
                    u_xmin + mask_u_ids[:, 0],
                    u_ymin + mask_u_ids[:, 1],
                    s=size2,
                    marker=">",
                    label="u_sten_hx_eq4",
                    color="cornflowerblue",
                )
                mask_u_ids = torch.argwhere(masks.u_sten_hx_gt6.squeeze())
                a.scatter(
                    u_xmin + mask_u_ids[:, 0],
                    u_ymin + mask_u_ids[:, 1],
                    s=size2,
                    marker=">",
                    label="u_sten_hx_gt6",
                    color="navy",
                )

                v_xmin, v_ymin = 0, -0.5
                mask_v_ids = torch.argwhere(masks.v_sten_hy_eq2.squeeze())
                a.scatter(
                    v_xmin + mask_v_ids[:, 0],
                    v_ymin + mask_v_ids[:, 1],
                    s=size2,
                    marker="^",
                    label="v_sten_hy_eq2",
                    color="gold",
                )
                mask_v_ids = torch.argwhere(masks.v_sten_hy_eq4.squeeze())
                a.scatter(
                    v_xmin + mask_v_ids[:, 0],
                    v_ymin + mask_v_ids[:, 1],
                    s=size2,
                    marker="^",
                    label="v_sten_hy_eq4",
                    color="darkorange",
                )
                mask_v_ids = torch.argwhere(masks.v_sten_hy_gt6.squeeze())
                a.scatter(
                    v_xmin + mask_v_ids[:, 0],
                    v_ymin + mask_v_ids[:, 1],
                    s=size2,
                    marker="^",
                    label="v_sten_hy_gt6",
                    color="firebrick",
                )
            elif stencil_var == "w":
                u_xmin, u_ymin = -0.5, 0
                mask_u_ids = torch.argwhere(masks.u_sten_wy_eq2.squeeze())
                a.scatter(
                    u_xmin + mask_u_ids[:, 0],
                    u_ymin + mask_u_ids[:, 1],
                    s=size2,
                    marker=">",
                    label="u_sten_wy_eq2",
                    color="lightblue",
                )
                mask_u_ids = torch.argwhere(masks.u_sten_wy_eq4.squeeze())
                a.scatter(
                    u_xmin + mask_u_ids[:, 0],
                    u_ymin + mask_u_ids[:, 1],
                    s=size2,
                    marker=">",
                    label="u_sten_wy_eq4",
                    color="cornflowerblue",
                )
                mask_u_ids = torch.argwhere(masks.u_sten_wy_gt6.squeeze())
                a.scatter(
                    u_xmin + mask_u_ids[:, 0],
                    u_ymin + mask_u_ids[:, 1],
                    s=size2,
                    marker=">",
                    label="u_sten_wy_gt6",
                    color="navy",
                )

                v_xmin, v_ymin = 0, -0.5
                mask_v_ids = torch.argwhere(masks.v_sten_wx_eq2.squeeze())
                a.scatter(
                    v_xmin + mask_v_ids[:, 0],
                    v_ymin + mask_v_ids[:, 1],
                    s=size2,
                    marker="^",
                    label="v_sten_wx_eq2",
                    color="gold",
                )
                mask_v_ids = torch.argwhere(masks.v_sten_wx_eq4.squeeze())
                a.scatter(
                    v_xmin + mask_v_ids[:, 0],
                    v_ymin + mask_v_ids[:, 1],
                    s=size2,
                    marker="^",
                    label="v_sten_wx_eq4",
                    color="darkorange",
                )
                mask_v_ids = torch.argwhere(masks.v_sten_wx_gt6.squeeze())
                a.scatter(
                    v_xmin + mask_v_ids[:, 0],
                    v_ymin + mask_v_ids[:, 1],
                    s=size2,
                    marker="^",
                    label="v_sten_wx_gt6",
                    color="firebrick",
                )

            f.legend(loc="upper left")
            plt.setp(a.get_xticklabels(), visible=False)
            plt.setp(a.get_yticklabels(), visible=False)
