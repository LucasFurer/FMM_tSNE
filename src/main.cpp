#include <FMM_tSNE/app.h>
#include <FMM_tSNE/cameras/two_D_camera.h>
#include <FMM_tSNE/tsne/tests/tsne_test.h>
#include <FMM_tSNE/tsne/tsne_buffers.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

int main()
{
    App app{};

    TsneTest tsne_test{};
    // tsne_test.errorTimestepTSNE("MNIST_digits", // data_set
    //                             10000,          // data_size
    //                             30.0f,          // perplexity
    //                             1000,           // iteration_amount
    //                             0.6f,           // theta
    //                             1.0f,           // cell_size
    //                             216308u         // seed
    // );

    // tsne_test.calculationtimeThetaTSNE("MNIST_digits",                                           // data_set
    //                                    10000,                                                    // data_size
    //                                    30.0f,                                                    // perplexity
    //                                    20,                                                       // iteration_amount
    //                                    std::vector<double>{0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0}, // thetas
    //                                    std::vector<double>{0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0}, // cell_sizes
    //                                    216308u                                                   // seed
    // );

    // tsne_test.errorThetaTSNE("MNIST_digits",                                           // data_set
    //                          1000,                                                     // data_size
    //                          30.0f,                                                    // perplexity
    //                          200,                                                      // iteration_amount
    //                          std::vector<float>{0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0},  // thetas
    //                          std::vector<double>{0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0}, // cell_sizes
    //                          216308u                                                   // seed
    // );

    // tsne_test.calculationtimeErrorTSNE("MNIST_digits",                                           // data_set
    //                                    70000,                                                    // data_size
    //                                    30.0f,                                                    // perplexity
    //                                    10,                                                       // iteration_amount
    //                                    std::vector<double>{0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0}, // thetas
    //                                    // std::vector<double>{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    //                                    // 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5}, // thetas
    //                                    std::vector<double>{0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0}, // cell_sizes
    //                                    // std::vector<double>{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
    //                                    // 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5}, // cell_sizes
    //                                    216308u // seed
    // );

    // comparing FMM to BH with high theta and looking at the cost
    // tsne_test.costTimestepTSNE("MNIST_digits", // data_set
    //                            70000,          // data_size
    //                            30.0f,          // perplexity
    //                            1001,           // iteration_amount
    //                            0.5,            // min_theta
    //                            3.0,            // max_theta
    //                            1.0,            // cell_size
    //                            216308u,        // seed
    //                            "BH" // method: "FMM_SYM_MORTON", "FMM_MORTON", "FMM", "PM", "BH", "BHRMP", "naive"
    // );

    // tsne_test.calculationtimeCostTSNE                              // looking at the final cost
    //     ("mice_brain_cells",                                       // data_set
    //      1000000,                                                  // data_size
    //      30.0f,                                                    // perplexity
    //      1001,                                                     // iteration_amount
    //      std::vector<double>{0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0}, // thetas
    //      std::vector<double>{0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0}, // cell_sizes
    //      216308u,                                                  // seed: 4523u
    //      // method: "FMMfast_SYM_MORTON", "FMM_SYM_MORTON", "FMM_MORTON", "FMM", "PM", "BH", "BHRMP", "naive"
    //      "PM");

    // tsne_test.calculationtimeError_MAXpoints_TSNE
    //(
    //     "MNIST_digits", // data_set
    //     70000, // data_size
    //     30.0f, // perplexity
    //     1001, // iteration_amount
    //     std::vector<double>{0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0}, // thetas
    //     std::vector<int>{1, 2, 4, 8, 16, 32, 64}, // max number of points in leaf nodes
    //     216308u, // seed: 4523u
    //     "FMM_SYM_MORTON" // method: "FMM_SYM_MORTON", "FMM_MORTON", "FMM", "PM", "BH", "BHRMP", "naive"
    //);

    // tsne_test.time_accuracy_test("MNIST_digits", // data_set
    //                              1000,           // data_size
    //                              30.0f,          // perplexity
    //                              1000,           // iteration_amount
    //                              10,             // repeat this many times
    //                              0.5f,           // theta
    //                              0.5f,           // cell_size
    //                              216308u         // seed
    // );

    // tsne_test.errorTimestepTSNE_Zaccum("MNIST_digits", // data_set
    //                                    10000,          // data_size
    //                                    30.0f,          // perplexity
    //                                    1000,           // iteration_amount
    //                                    1.0f,           // theta
    //                                    1.0f,           // cell_size
    //                                    216308u         // seed
    // );

    std::cout << "done with tests\n";
    // -------------------------------

    // put stuff into the app
    glm::mat4 model = glm::scale(glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 0.0f)), glm::vec3(1.0f));

    Shader shader_tsne = Shader((std::filesystem::current_path().string() + "/shaders/shader_tsne.vs").c_str(),
                                (std::filesystem::current_path().string() + "/shaders/shader_tsne.fs").c_str());

    TSNE_buffers tsne(0.7,             // min_theta
                      0.7,             // max_theta
                      1.0,             // cell_size
                      "MNIST_fashion", // data_set: "MNIST_digits", "MNIST_fashion", "mice_brain_cells", "CIFAR10"
                      70000,           // data_size
                      30.0f,           // perplexity
                      216308u          // seed: 216308u, 592340823u, 4523u, 296343u
    );
    tsne.nBodySelect = "FMMfast_SYM_MORTON";
    // tsne.nBodySelect = "PM";

    Renderable points_renderable(GL_POINTS, model, tsne.embeddedBuffer, &shader_tsne, nullptr);

    Shader shader_lines = Shader((std::filesystem::current_path().string() + "/shaders/shader_lines.vs").c_str(),
                                 (std::filesystem::current_path().string() + "/shaders/shader_lines.fs").c_str());

    Renderable tree_structure_renderable(GL_LINES, model, tsne.nodeBuffer, &shader_lines, nullptr);
    Renderable forces_renderable(GL_LINES, model, tsne.forceBuffer, &shader_lines, nullptr);

    std::vector<Renderable> renderables{points_renderable, tree_structure_renderable, forces_renderable};

    TwoDCamera camera = TwoDCamera(glm::vec3(0.0f, 0.0f, -800.0f),
                                   glm::vec3(0.0f, 1.0f, 0.0f),
                                   90.0f,
                                   0.0f,
                                   glm::vec3(0.0f, 0.0f, -1.0f),
                                   2.0f,
                                   0.1f,
                                   200.0f,
                                   0.001f,
                                   1000.0f,
                                   false,
                                   &App::screen_width_,
                                   &App::screen_height_);

    Scene scene = Scene("tsne", &camera, renderables);

    app.scenes_[scene.sceneName] = &scene;

    // scene names
    for (const std::pair<const std::string, Scene*>& scene : app.scenes_)
    {
        app.scene_names_.push_back(scene.second->sceneName);
    }
    for (int i = 0; i < app.scene_names_.size(); i++)
    {
        if (app.scene_names_[i] == app.current_scene_name_)
        {
            app.current_scene_nameIndex_ = i;
        }
    }

    while (!app.should_close())
    {
        app.run({.tsne = tsne});
    }

    return 0;
}
