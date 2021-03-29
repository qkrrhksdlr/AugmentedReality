#include "solution.h"
#include <opencv2/opencv.hpp>

int main()
{
	// Reference Image Feature Database 
	std::vector<cv::Mat> DataBase;
	reference(DataBase);

	// Feature Matching
	cv::String Canon_path("cd_covers/Canon/*.jpg");
	std::vector<cv::String> Canon;

	cv::String Droid_path("cd_covers/Droid/*.jpg");
	std::vector<cv::String> Droid;

	cv::String E63_path("cd_covers/E63/*.jpg");
	std::vector<cv::String> E63;

	cv::String Palm_path("cd_covers/Palm/*.jpg");
	std::vector<cv::String> Palm;
	
	while (1)
	{
		int answer;
		std::cout << "           *** Choose Camera ***" << std::endl;
		std::cout << "{ Canon(1) / Droid(2) / E63(3) / Palm(4) / exit(0)}" << std::endl;
		std::cin >> answer;

		if (answer == 0)
		{
			std::cout << "Finished" << std::endl;
			return -1;
		}

		switch (answer)
		{
			case 1 :
				matching(Canon_path, Canon, DataBase);
				break;

			case 2 :
				matching(Droid_path, Droid, DataBase);
				break;

			case 3 :
				matching(E63_path, E63, DataBase);
				break;

			case 4 :
				matching(Palm_path, Palm, DataBase);
				break;

			default :
				std::cout << "Wrong Answer. Try Again" << std::endl << std::endl;
				continue;
		}
	}

	return 0;
}