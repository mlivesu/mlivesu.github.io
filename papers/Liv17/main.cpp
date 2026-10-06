#include <QApplication>

#include <cinolib/meshes/drawable_trimesh.h>
#include <cinolib/meshes/drawable_tetmesh.h>
#include <cinolib/meshes/drawable_polygonmesh.h>
#include <cinolib/meshes/abstract_mesh.h>
#include <cinolib/vector_field.h>
#include <cinolib/scalar_field.h>
#include <cinolib/gradient.h>
#include <cinolib/bfs.h>
#include <cinolib/heat_flow.h>
#include <cinolib/color.h>
#include <cinolib/gl/glcanvas.h>
#include <cinolib/drawable_isocontour.h>
#include <cinolib/drawable_isosurface.h>

// this implementation assumes a triangle mesh if TRI is enabled, a tetrahedral mesh otherwise
//#define TRI

// ALGORITHM PARAMETERS
#define LAMBDA_TRI       0.2
#define LAMBDA_TET       0.01
#define SMOOTHING_PASSES 10


template <class Mesh>
void smooth_discrete_hyper_surface(Mesh & m)
{
    using namespace cinolib;

    // STEP ONE: heat flow
    std::vector<uint> heat_sources;
    for(uint vid=0; vid<m.num_verts(); ++vid)
    {
        bool has_A = false;
        bool has_B = false;
        for(uint pid : m.adj_v2p(vid))
        {
            if (m.poly_data(pid).label == 0) has_A = true; else
            if (m.poly_data(pid).label == 1) has_B = true;
        }
        if (has_A && has_B) heat_sources.push_back(vid);
    }
    double t = std::pow(m.edge_avg_length(),2.0);
    ScalarField u = heat_flow(m, heat_sources, t, COTANGENT);
    u.normalize_in_01();
    u.copy_to_mesh(m);
    u.serialize("u");
    std::cout << "heat flow computed" << std::endl;

    VectorField field = VectorField(m.num_polys());
    field = gradient_matrix(m) * u;
    field.serialize("u_gradient");

    std::cout << "u gradient computed" << std::endl;

    // STEP TWO: flip the gradient of one of the regions
    for(uint pid=0; pid<m.num_polys(); ++pid)
    {
        if (m.poly_data(pid).label == 1) field.set(pid, -field.vec_at(pid));
    }
    field.normalize();
    field.serialize("X");

    std::cout << "X field generated" << std::endl;

    // STEP THREE: smooth the resulting gradient
    for(uint i=0; i<SMOOTHING_PASSES; ++i)
    for(uint pid=0; pid<m.num_polys(); ++pid)
    {
        vec3d avg_g = field.vec_at(pid);
        for(uint nbr : m.adj_p2p(pid))
        {
            avg_g += field.vec_at(nbr);
        }
        avg_g /= static_cast<double>(m.adj_p2p(pid).size()+1);
        avg_g.normalize();
        field.set(pid,avg_g);
    }
    field.normalize();
    field.serialize("X_prime");

    std::cout << "smoothed X field generated" << std::endl;

    // STEP FOUR: find the scalar field corresponding to it
    std::vector<Eigen::Triplet<double>> entries = laplacian_matrix_entries(m, COTANGENT);
    Eigen::VectorXd div = gradient_matrix(m).transpose() * field;
    Eigen::SparseMatrix<double> L(m.num_verts()+heat_sources.size(), m.num_verts());
    Eigen::VectorXd rhs(m.num_verts()+heat_sources.size());
    for(uint vid=0; vid<m.num_verts(); ++vid) rhs[vid] = div[vid];
#ifdef TRI
    double lambda = LAMBDA_TRI;
#else
    double lambda = LAMBDA_TET;
#endif
    for(uint i=0; i<heat_sources.size(); ++i)
    {
        uint vid = heat_sources.at(i);
        entries.push_back(Entry(m.num_verts()+i, vid, lambda * 1.0));
        rhs[m.num_verts()+i] = lambda * 0.0;
    }
    L.setFromTriplets(entries.begin(), entries.end());
    ScalarField phi;
    solve_least_squares(-L, rhs, phi);
    phi.copy_to_mesh(m);

    phi.normalize_in_01();
    phi.serialize("phi");

    std::cout << "phi scalar function computed" << std::endl;
}


int main(int argc, char **argv)
{
    using namespace cinolib;

    QApplication a(argc, argv);

    if (true || argc != 4)
    {
        std::cout << "\nDemo software for Discrete                                                                     " << std::endl;
        std::cout << "usage:\n\tdiscrete_boundary_smoothing mesh seed_r1 seed_r2             " << std::endl;
        std::cout << "\t mesh  : either a triangle mesh (.OBJ,.OFF) or a tetmesh (.TET,.MESH)" << std::endl;
        std::cout << "\t seeds : id of two faces/tets used to generate a fake segmentation with a discrete boundary to be smoothed" << std::endl;
        return -1;
    }

#ifdef TRI
    DrawableTrimesh<> m(argv[1]);
#else
    DrawableTetmesh<> m(argv[1]);
#endif
    std::cout << "mesh loaded" << std::endl;

    uint source_A = atoi(argv[2]);
    uint source_B = atoi(argv[3]);

    // PRE-PROCESSING: generate a binary labeling
    std::vector<double> dist_A;
    std::vector<double> dist_B;
    bfs_exahustive_on_dual(m, source_A, dist_A);
    bfs_exahustive_on_dual(m, source_B, dist_B);
    std::cout << "labeling generated" << std::endl;

    ScalarField labeling(m.num_polys());

    for(uint pid=0; pid<m.num_polys(); ++pid)
    {
        int label = (dist_A.at(pid) <= dist_B.at(pid)) ? 0 : 1;
        m.poly_data(pid).label = label;
        m.poly_data(pid).color = (label==0) ? Color::PASTEL_YELLOW() : Color::PASTEL_CYAN();
        labeling[pid] = label;
#ifndef TRI // just for visualization
        for(uint fid : m.adj_p2f(pid))
        {
            m.face_data(fid).label = label;
            m.face_data(fid).color = (label==0) ? Color::PASTEL_YELLOW() : Color::PASTEL_CYAN();
        }
#endif
    }
    labeling.serialize("labeling");
    std::cout << "labeling attached to mesh" << std::endl;

    smooth_discrete_hyper_surface(m);

#ifdef TRI
    m.show_face_color();
#endif
    m.show_face_wireframe(true);
    m.updateGL();

#ifdef TRI
    DrawableIsocontour smoothed_boundary(m,0.0);
#else
    DrawableIsosurface<> smoothed_boundary(m,0.0);
    smoothed_boundary.export_as_trimesh().save("smoothed_hypersurface.obj");
#endif

    GLcanvas gui;
    gui.push_obj(&m);
#ifndef TRI
    gui.push_obj(&smoothed_boundary,false);
#endif
    gui.show();

    return a.exec();
}
