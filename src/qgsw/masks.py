# ruff: noqa
import torch
import torch.nn.functional as F


class Masks:
    """
    Masks for staggered mesh used in Shallow-water discretization.
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

    def __init__(self, mask_hmesh: torch.Tensor):
        """
        Computes automatically several masks given the binary mask on the h mesh.
        as well as the irregulare boundary points
        Example of mask:
            Given the following 4_x3 domain:

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
            i.e. on the rectangle boundary, and o are irregular boundary points.


        Parameters
            - mask_hmesh: float/double Tensor, shape (nx, ny), binary values
        """
        mtype = mask_hmesh.dtype
        mshape = mask_hmesh.shape

        self.h = mask_hmesh.reshape((1,) * (4 - len(mshape)) + mshape)
        self.u = (
            F.avg_pool2d(self.h, (2, 1), stride=(1, 1), padding=(1, 0)) > 3 / 4
        )
        self.v = (
            F.avg_pool2d(self.h, (1, 2), stride=(1, 1), padding=(0, 1)) > 3 / 4
        )
        self.w = (
            F.avg_pool2d(self.h, (2, 2), stride=(1, 1), padding=(1, 1)) > 1 / 8
        )
        self.psi = (
            F.avg_pool2d(self.h, (2, 2), stride=(1, 1), padding=(1, 1)) > 7 / 8
        )

        self.not_h = torch.logical_not(self.h.type(torch.bool))
        self.not_u = torch.logical_not(self.u)
        self.not_v = torch.logical_not(self.v)
        self.not_w = torch.logical_not(self.w)
        self.not_psi = torch.logical_not(self.psi)

        # Vorticity vertical and horizontal boundary
        w_1neighbor_v = (
            F.avg_pool2d(
                self.v.type(mtype),
                (2, 1),
                stride=(1, 1),
                padding=(1, 0),
                divisor_override=1,
            )
            == 1
        )
        w_1neighbor_u = (
            F.avg_pool2d(
                self.u.type(mtype),
                (1, 2),
                stride=(1, 1),
                padding=(0, 1),
                divisor_override=1,
            )
            == 1
        )
        self.w_vertical_bound = self.w * torch.logical_and(
            w_1neighbor_v, torch.logical_not(w_1neighbor_u)
        )
        self.w_horizontal_bound = self.w * torch.logical_and(
            w_1neighbor_u, torch.logical_not(w_1neighbor_v)
        )
        self.w_cornerout_bound = self.w * torch.logical_and(
            w_1neighbor_u, w_1neighbor_v
        )
        self.w_valid = self.w * torch.logical_and(
            torch.logical_and(
                torch.logical_not(self.w_vertical_bound),
                torch.logical_not(self.w_horizontal_bound),
            ),
            torch.logical_not(self.w_cornerout_bound),
        )

        # Irregular boundary indices
        self.psi_irrbound_xids, self.psi_irrbound_yids = torch.where(
            torch.logical_and(
                self.not_psi[0, 0, 1:-1, 1:-1],
                F.avg_pool2d(self.psi.type(mtype), (3, 3), stride=(1, 1))[0, 0]
                > 1 / 18,
            )
        )

        # h-stencil in direction x on u-mesh
        self.u_sten_hx_eq2 = torch.logical_and(
            torch.logical_and(
                F.avg_pool2d(
                    self.h.type(mtype), (2, 1), stride=(1, 1), padding=(1, 0)
                )
                > 3 / 4,
                F.avg_pool2d(
                    self.h.type(mtype), (4, 1), stride=(1, 1), padding=(2, 0)
                )
                < 7 / 8,
            ),
            self.u,
        )
        self.u_sten_hx_gt4 = torch.logical_and(
            torch.logical_not(self.u_sten_hx_eq2), self.u
        )
        self.u_sten_hx_eq4 = torch.logical_and(
            F.avg_pool2d(
                self.h.type(mtype), (6, 1), stride=(1, 1), padding=(3, 0)
            )
            < 11 / 12,
            self.u_sten_hx_gt4,
        )
        self.u_sten_hx_gt6 = torch.logical_and(
            torch.logical_not(self.u_sten_hx_eq4), self.u_sten_hx_gt4
        )

        # w-stencil in direction y on u-mesh
        self.u_sten_wy_eq2 = torch.logical_and(
            torch.logical_and(
                F.avg_pool2d(
                    self.w.type(mtype), (1, 2), stride=(1, 1), padding=(0, 0)
                )
                > 3 / 4,
                F.avg_pool2d(
                    self.w.type(mtype), (1, 4), stride=(1, 1), padding=(0, 1)
                )
                < 7 / 8,
            ),
            self.u,
        )
        self.u_sten_wy_gt4 = torch.logical_and(
            torch.logical_not(self.u_sten_wy_eq2), self.u
        )
        self.u_sten_wy_eq4 = torch.logical_and(
            F.avg_pool2d(
                self.w.type(mtype), (1, 6), stride=(1, 1), padding=(0, 2)
            )
            < 11 / 12,
            self.u_sten_wy_gt4,
        )
        self.u_sten_wy_gt6 = torch.logical_and(
            torch.logical_not(self.u_sten_wy_eq4), self.u_sten_wy_gt4
        )

        # h-stencil in direction y on v-mesh
        self.v_sten_hy_eq2 = torch.logical_and(
            torch.logical_and(
                F.avg_pool2d(
                    self.h.type(mtype), (1, 2), stride=(1, 1), padding=(0, 1)
                )
                > 3 / 4,
                F.avg_pool2d(
                    self.h.type(mtype), (1, 4), stride=(1, 1), padding=(0, 2)
                )
                < 7 / 8,
            ),
            self.v,
        )
        self.v_sten_hy_gt4 = torch.logical_and(
            torch.logical_not(self.v_sten_hy_eq2), self.v
        )
        self.v_sten_hy_eq4 = torch.logical_and(
            F.avg_pool2d(
                self.h.type(mtype), (1, 6), stride=(1, 1), padding=(0, 3)
            )
            < 11 / 12,
            self.v_sten_hy_gt4,
        )
        self.v_sten_hy_gt6 = torch.logical_and(
            torch.logical_not(self.v_sten_hy_eq4), self.v_sten_hy_gt4
        )

        # w-stencil in direction x on v-mesh
        self.v_sten_wx_eq2 = torch.logical_and(
            torch.logical_and(
                F.avg_pool2d(
                    self.w.type(mtype), (2, 1), stride=(1, 1), padding=(0, 0)
                )
                > 3 / 4,
                F.avg_pool2d(
                    self.w.type(mtype), (4, 1), stride=(1, 1), padding=(1, 0)
                )
                < 7 / 8,
            ),
            self.v,
        )
        self.v_sten_wx_gt4 = torch.logical_and(
            torch.logical_not(self.v_sten_wx_eq2), self.v
        )
        self.v_sten_wx_eq4 = torch.logical_and(
            F.avg_pool2d(
                self.w.type(mtype), (6, 1), stride=(1, 1), padding=(2, 0)
            )
            < 11 / 12,
            self.v_sten_wx_gt4,
        )
        self.v_sten_wx_gt6 = torch.logical_and(
            torch.logical_not(self.v_sten_wx_eq4), self.v_sten_wx_gt4
        )

        # convert masks to correct data type
        self.h = self.h.type(mtype)
        self.u = self.u.type(mtype)
        self.v = self.v.type(mtype)
        self.w = self.w.type(mtype)
        self.w_vertical_bound = self.w_vertical_bound.type(mtype)
        self.w_horizontal_bound = self.w_horizontal_bound.type(mtype)
        self.w_valid = self.w_valid.type(mtype)
        self.psi = self.psi.type(mtype)
        self.not_h = self.not_h.type(mtype)
        self.not_u = self.not_u.type(mtype)
        self.not_v = self.not_v.type(mtype)
        self.not_w = self.not_w.type(mtype)
        self.not_psi = self.not_psi.type(mtype)

        self.u_sten_hx_eq2 = self.u_sten_hx_eq2.type(mtype)
        self.u_sten_hx_eq4 = self.u_sten_hx_eq4.type(mtype)
        self.u_sten_hx_gt4 = self.u_sten_hx_gt4.type(mtype)
        self.u_sten_hx_gt6 = self.u_sten_hx_gt6.type(mtype)

        self.u_sten_wy_eq2 = self.u_sten_wy_eq2.type(mtype)
        self.u_sten_wy_eq4 = self.u_sten_wy_eq4.type(mtype)
        self.u_sten_wy_gt4 = self.u_sten_wy_gt4.type(mtype)
        self.u_sten_wy_gt6 = self.u_sten_wy_gt6.type(mtype)

        self.v_sten_hy_eq2 = self.v_sten_hy_eq2.type(mtype)
        self.v_sten_hy_eq4 = self.v_sten_hy_eq4.type(mtype)
        self.v_sten_hy_gt4 = self.v_sten_hy_gt4.type(mtype)
        self.v_sten_hy_gt6 = self.v_sten_hy_gt6.type(mtype)

        self.v_sten_wx_eq2 = self.v_sten_wx_eq2.type(mtype)
        self.v_sten_wx_eq4 = self.v_sten_wx_eq4.type(mtype)
        self.v_sten_wx_gt4 = self.v_sten_wx_gt4.type(mtype)
        self.v_sten_wx_gt6 = self.v_sten_wx_gt6.type(mtype)


if __name__ == "__main__":
    import numpy as np
    import matplotlib

    matplotlib.rcParams.update({"font.size": 24})
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
            a.mesh()

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
